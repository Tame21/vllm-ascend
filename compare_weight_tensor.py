#!/usr/bin/env python3
"""Compare one tensor stored in two model checkpoints.

Supported inputs:
  * A single .safetensors, .bin, .pt, or .pth file
  * A Hugging Face model directory, including sharded checkpoints

Install dependencies:
  pip install torch safetensors

This tool compares the values physically stored in the checkpoint. Integer or
packed quantized tensors are cast to float64 for metric calculation, but are
NOT automatically dequantized with a model-specific quantization algorithm.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: install PyTorch with `pip install torch`.") from exc


TORCH_SUFFIXES = {".bin", ".pt", ".pth"}
SAFE_SUFFIX = ".safetensors"


def _safe_open(path: Path):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit(
            "Reading .safetensors requires `pip install safetensors`."
        ) from exc
    return safe_open(str(path), framework="pt", device="cpu")


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise ValueError("PyTorch checkpoint is not a dictionary.")
    for wrapper in ("state_dict", "model", "module"):
        nested = obj.get(wrapper)
        if isinstance(nested, dict) and any(
            isinstance(v, torch.Tensor) for v in nested.values()
        ):
            obj = nested
            break
    return {k: v for k, v in obj.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}


def _load_torch_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        obj = torch.load(str(path), map_location="cpu")
    return _unwrap_state_dict(obj)


class Checkpoint:
    def __init__(self, source: str):
        self.path = Path(source).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.path}")
        self.weight_map: dict[str, Path] | None = None
        self.files = self._discover_files()

    def _discover_files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]

        indexes = sorted(self.path.glob("*.index.json"))
        for index in indexes:
            try:
                data = json.loads(index.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            raw_map = data.get("weight_map")
            if isinstance(raw_map, dict):
                self.weight_map = {
                    str(k): self.path / str(v) for k, v in raw_map.items()
                }
                return sorted(set(self.weight_map.values()))

        safe_files = sorted(self.path.glob(f"*{SAFE_SUFFIX}"))
        if safe_files:
            return safe_files
        torch_files = sorted(
            p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in TORCH_SUFFIXES
        )
        if torch_files:
            return torch_files
        raise FileNotFoundError(f"No supported weight files found under: {self.path}")

    def keys(self) -> list[str]:
        if self.weight_map is not None:
            return sorted(self.weight_map)
        result: list[str] = []
        for file in self.files:
            if file.suffix.lower() == SAFE_SUFFIX:
                with _safe_open(file) as handle:
                    result.extend(handle.keys())
            elif file.suffix.lower() in TORCH_SUFFIXES:
                result.extend(_load_torch_file(file).keys())
            else:
                raise ValueError(f"Unsupported checkpoint file: {file}")
        return sorted(set(result))

    def tensor(self, name: str) -> tuple[torch.Tensor, Path]:
        if self.weight_map is not None and name not in self.weight_map:
            self._raise_missing(name)
        candidates = [self.weight_map[name]] if self.weight_map is not None else self.files
        for file in candidates:
            if not file.exists():
                raise FileNotFoundError(f"Shard referenced by index does not exist: {file}")
            suffix = file.suffix.lower()
            if suffix == SAFE_SUFFIX:
                with _safe_open(file) as handle:
                    if name in handle.keys():
                        return handle.get_tensor(name), file
            elif suffix in TORCH_SUFFIXES:
                state = _load_torch_file(file)
                if name in state:
                    return state[name].detach().cpu(), file
            else:
                raise ValueError(f"Unsupported checkpoint file: {file}")
        self._raise_missing(name)
        raise AssertionError("unreachable")

    def _raise_missing(self, name: str) -> None:
        keys = self.keys()
        close = difflib.get_close_matches(name, keys, n=10, cutoff=0.25)
        contains = [key for key in keys if name.lower() in key.lower()][:10]
        suggestions = list(dict.fromkeys(close + contains))
        suffix = "\nClosest tensor names:\n  " + "\n  ".join(suggestions) if suggestions else ""
        raise KeyError(f"Tensor {name!r} was not found in {self.path}.{suffix}")


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int,
    sample_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"Shape mismatch: model A {tuple(a.shape)} vs model B {tuple(b.shape)}")
    if a.is_complex() or b.is_complex():
        raise ValueError("Complex tensors are not supported.")

    af = a.reshape(-1)
    bf = b.reshape(-1)
    total = af.numel()
    sums = {key: 0.0 for key in ("a", "b", "aa", "bb", "ab", "abs", "sq")}
    valid = close = same_sign = 0
    max_abs = 0.0

    with torch.no_grad():
        for start in range(0, total, chunk_size):
            x = af[start : start + chunk_size].to(torch.float64)
            y = bf[start : start + chunk_size].to(torch.float64)
            mask = torch.isfinite(x) & torch.isfinite(y)
            if not bool(mask.any()):
                continue
            x, y = x[mask], y[mask]
            d = x - y
            n = x.numel()
            valid += n
            sums["a"] += x.sum().item()
            sums["b"] += y.sum().item()
            sums["aa"] += torch.dot(x, x).item()
            sums["bb"] += torch.dot(y, y).item()
            sums["ab"] += torch.dot(x, y).item()
            sums["abs"] += d.abs().sum().item()
            sums["sq"] += torch.dot(d, d).item()
            max_abs = max(max_abs, d.abs().max().item())
            close += torch.isclose(x, y, rtol=rtol, atol=atol).sum().item()
            same_sign += (torch.sign(x) == torch.sign(y)).sum().item()

    if valid == 0:
        raise ValueError("No pair of finite values is available for comparison.")

    norm_a, norm_b = math.sqrt(sums["aa"]), math.sqrt(sums["bb"])
    cosine = sums["ab"] / (norm_a * norm_b) if norm_a and norm_b else math.nan
    mean_a, mean_b = sums["a"] / valid, sums["b"] / valid
    var_a = max(0.0, sums["aa"] - valid * mean_a * mean_a)
    var_b = max(0.0, sums["bb"] - valid * mean_b * mean_b)
    covariance = sums["ab"] - valid * mean_a * mean_b
    pearson = covariance / math.sqrt(var_a * var_b) if var_a and var_b else math.nan
    mse = sums["sq"] / valid
    rel_l2 = math.sqrt(sums["sq"]) / norm_a if norm_a else math.nan
    snr_db = 20.0 * math.log10(norm_a / math.sqrt(sums["sq"])) if sums["sq"] and norm_a else math.inf

    # Evenly spaced samples keep distribution metrics bounded in memory.
    count = min(total, sample_size)
    if count:
        indices = torch.linspace(0, total - 1, count, dtype=torch.int64)
        sx = af[indices].to(torch.float64)
        sy = bf[indices].to(torch.float64)
        finite = torch.isfinite(sx) & torch.isfinite(sy)
        abs_sample = (sx[finite] - sy[finite]).abs()
        quantiles = {
            str(q): torch.quantile(abs_sample, q).item()
            for q in (0.5, 0.9, 0.99)
        } if abs_sample.numel() else {}
    else:
        quantiles = {}

    return {
        "numel": total,
        "finite_pairs": valid,
        "non_finite_pairs": total - valid,
        "cosine_similarity": _json_number(cosine),
        "pearson_correlation": _json_number(pearson),
        "mae": sums["abs"] / valid,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "max_absolute_error": max_abs,
        "relative_l2_error_vs_a": _json_number(rel_l2),
        "snr_db_vs_a": _json_number(snr_db),
        "close_fraction": close / valid,
        "same_sign_fraction": same_sign / valid,
        "l2_norm_a": norm_a,
        "l2_norm_b": norm_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "abs_error_quantiles_sampled": quantiles,
        "distribution_sample_size": min(total, sample_size),
    }


def _format_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"Tensor A : {report['tensor_a']}",
        f"Tensor B : {report['tensor_b']}",
        f"Shape    : {report['shape']}",
        f"Dtype    : {report['dtype_a']} vs {report['dtype_b']}",
        f"Storage  : {report['shard_a']} vs {report['shard_b']}",
        "",
        f"Cosine similarity      : {metrics['cosine_similarity']}",
        f"Pearson correlation    : {metrics['pearson_correlation']}",
        f"MAE / RMSE / MSE       : {metrics['mae']:.8g} / {metrics['rmse']:.8g} / {metrics['mse']:.8g}",
        f"Max absolute error     : {metrics['max_absolute_error']:.8g}",
        f"Relative L2 error vs A : {metrics['relative_l2_error_vs_a']}",
        f"SNR vs A (dB)          : {metrics['snr_db_vs_a']}",
        f"Close fraction         : {metrics['close_fraction']:.6%}",
        f"Same-sign fraction     : {metrics['same_sign_fraction']:.6%}",
        f"L2 norm A / B          : {metrics['l2_norm_a']:.8g} / {metrics['l2_norm_b']:.8g}",
        f"Mean A / B             : {metrics['mean_a']:.8g} / {metrics['mean_b']:.8g}",
        f"Finite / total pairs   : {metrics['finite_pairs']} / {metrics['numel']}",
        f"Sampled |error| q50/90/99: {metrics['abs_error_quantiles_sampled']}",
    ]
    if report["raw_quantized_values"]:
        lines += [
            "",
            "NOTE: At least one tensor has an integer/quantized dtype. Metrics compare",
            "the stored values after a float cast; model-specific dequantization was not applied.",
        ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a named tensor from two model checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_a", help="First checkpoint file or model directory")
    parser.add_argument("model_b", help="Second checkpoint file or model directory")
    parser.add_argument("--layer", help="Tensor name used for both checkpoints")
    parser.add_argument("--layer-a", help="Tensor name in model A (overrides --layer)")
    parser.add_argument("--layer-b", help="Tensor name in model B (defaults to A's name)")
    parser.add_argument("--list-layers", action="store_true", help="List tensor names and exit")
    parser.add_argument("--filter", default="", help="Substring filter used with --list-layers")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for close_fraction")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for close_fraction")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Elements processed per metric chunk")
    parser.add_argument("--sample-size", type=int, default=1_000_000, help="Max samples for error quantiles")
    parser.add_argument("--json", dest="json_path", help="Also write the full report as JSON")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checkpoint_a, checkpoint_b = Checkpoint(args.model_a), Checkpoint(args.model_b)
        if args.list_layers:
            for label, checkpoint in (("A", checkpoint_a), ("B", checkpoint_b)):
                print(f"[{label}] {checkpoint.path}")
                for key in checkpoint.keys():
                    if args.filter.lower() in key.lower():
                        print(key)
            return 0

        layer_a = args.layer_a or args.layer
        layer_b = args.layer_b or layer_a
        if not layer_a:
            raise ValueError("Specify --layer (or --layer-a), or use --list-layers.")
        if args.chunk_size <= 0 or args.sample_size < 0:
            raise ValueError("--chunk-size must be positive and --sample-size non-negative.")

        tensor_a, shard_a = checkpoint_a.tensor(layer_a)
        tensor_b, shard_b = checkpoint_b.tensor(layer_b)
        report = {
            "model_a": str(checkpoint_a.path),
            "model_b": str(checkpoint_b.path),
            "tensor_a": layer_a,
            "tensor_b": layer_b,
            "shard_a": str(shard_a),
            "shard_b": str(shard_b),
            "shape": list(tensor_a.shape),
            "dtype_a": str(tensor_a.dtype),
            "dtype_b": str(tensor_b.dtype),
            "raw_quantized_values": not tensor_a.is_floating_point() or not tensor_b.is_floating_point(),
            "metrics": compare_tensors(
                tensor_a, tensor_b, args.chunk_size, args.sample_size, args.atol, args.rtol
            ),
        }
        print(_format_report(report))
        if args.json_path:
            output = Path(args.json_path).expanduser()
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nJSON report written to: {output.resolve()}")
        return 0
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
