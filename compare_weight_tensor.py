#!/usr/bin/env python3
"""Compare one tensor stored in two model checkpoints.

Supported inputs:
  * A single .safetensors, .bin, .pt, or .pth file
  * A Hugging Face model directory, including sharded checkpoints

Install dependencies:
  pip install torch safetensors

MXFP4 is automatically recognized in both OpenAI ``weight.blocks`` +
``weight.scales`` layout and Ascend ``weight`` + ``weight_scale`` layout. Other
integer or packed tensors are compared as stored unless a format is specified.
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
        self._keys_cache: list[str] | None = None
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
        if self._keys_cache is not None:
            return self._keys_cache
        if self.weight_map is not None:
            self._keys_cache = sorted(self.weight_map)
            return self._keys_cache
        result: list[str] = []
        for file in self.files:
            if file.suffix.lower() == SAFE_SUFFIX:
                with _safe_open(file) as handle:
                    result.extend(handle.keys())
            elif file.suffix.lower() in TORCH_SUFFIXES:
                result.extend(_load_torch_file(file).keys())
            else:
                raise ValueError(f"Unsupported checkpoint file: {file}")
        self._keys_cache = sorted(set(result))
        return self._keys_cache

    def has_tensor(self, name: str) -> bool:
        return name in self.keys()

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


def _compare_chunks(
    chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
    total: int,
    sample_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    sums = {key: 0.0 for key in ("a", "b", "aa", "bb", "ab", "abs", "sq")}
    valid = close = same_sign = 0
    max_abs = 0.0
    sample_parts: list[torch.Tensor] = []
    sample_stride = max(1, math.ceil(total / sample_size)) if sample_size else 0
    offset = 0

    with torch.no_grad():
        for raw_x, raw_y in chunks:
            x = raw_x.reshape(-1).to(torch.float64)
            y = raw_y.reshape(-1).to(torch.float64)
            if x.numel() != y.numel():
                raise ValueError("Internal error: comparison chunks have different sizes.")
            chunk_len = x.numel()
            if sample_stride:
                first = (-offset) % sample_stride
                if first < chunk_len:
                    sample_x = x[first::sample_stride]
                    sample_y = y[first::sample_stride]
                    sample_mask = torch.isfinite(sample_x) & torch.isfinite(sample_y)
                    sample_parts.append((sample_x[sample_mask] - sample_y[sample_mask]).abs().cpu())
            offset += chunk_len
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

    if sample_parts:
        abs_sample = torch.cat(sample_parts)[:sample_size]
        quantiles = {
            str(q): torch.quantile(abs_sample, q).item()
            for q in (0.5, 0.9, 0.99)
        }
    else:
        abs_sample = torch.empty(0)
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
        "distribution_sample_size": abs_sample.numel(),
    }


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
    af, bf = a.reshape(-1), b.reshape(-1)
    chunks = (
        (af[start : start + chunk_size], bf[start : start + chunk_size])
        for start in range(0, af.numel(), chunk_size)
    )
    return _compare_chunks(chunks, af.numel(), sample_size, atol, rtol)


FP4_E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _mxfp_scale(scales: torch.Tensor, mode: str) -> torch.Tensor:
    dtype_name = str(scales.dtype)
    if mode == "auto":
        if "e8m0" in dtype_name:
            mode = "e8m0"
        elif scales.dtype == torch.uint8:
            mode = "e8m0"
        elif scales.is_floating_point():
            mode = "float"
        else:
            mode = "exponent"
    if mode == "float":
        return scales.to(torch.float64)
    if mode == "e8m0":
        raw = scales.view(torch.uint8) if "e8m0" in dtype_name else scales.to(torch.uint8)
        exponent = raw.to(torch.int32) - 127
    else:
        exponent = scales.to(torch.int32)
    return torch.ldexp(torch.ones_like(exponent, dtype=torch.float64), exponent)


def _mxfp4_layout(
    blocks: torch.Tensor, scales: torch.Tensor, block_size: int
) -> tuple[str, tuple[int, ...]]:
    if blocks.ndim < 1 or scales.ndim < 1:
        raise ValueError("MXFP4 weight and scale tensors must not be scalars.")
    # OpenAI/gpt-oss: [..., groups, block_size/2] plus scales [..., groups].
    if tuple(blocks.shape[:-1]) == tuple(scales.shape):
        if blocks.shape[-1] * 2 != block_size:
            raise ValueError(
                f"Packed block has {blocks.shape[-1]} bytes; expected {block_size // 2}."
            )
        return "grouped", (*blocks.shape[:-2], blocks.shape[-2] * block_size)
    # Ascend ModelSlim/vLLM: [..., K/2] weight plus [..., K/block_size] scales.
    if (
        tuple(blocks.shape[:-1]) == tuple(scales.shape[:-1])
        and blocks.shape[-1] * 2 == scales.shape[-1] * block_size
    ):
        return "ascend_row", (*blocks.shape[:-1], blocks.shape[-1] * 2)
    raise ValueError(
        f"MXFP4 weight/scale shape mismatch: {tuple(blocks.shape)} vs {tuple(scales.shape)}; "
        f"block size is {block_size}."
    )


def compare_mxfp4(
    blocks_a: torch.Tensor,
    scales_a: torch.Tensor,
    blocks_b: torch.Tensor,
    scales_b: torch.Tensor,
    block_size: int,
    scale_mode_a: str,
    scale_mode_b: str,
    chunk_size: int,
    sample_size: int,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    layout_a, shape_a = _mxfp4_layout(blocks_a, scales_a, block_size)
    layout_b, shape_b = _mxfp4_layout(blocks_b, scales_b, block_size)
    if shape_a != shape_b:
        raise ValueError(f"Dequantized shape mismatch: model A {shape_a} vs model B {shape_b}")
    if layout_a != layout_b:
        raise ValueError(f"MXFP4 layouts differ: model A {layout_a} vs model B {layout_b}")

    rows_per_chunk = max(1, chunk_size // block_size)
    if layout_a == "grouped":
        bytes_per_row = block_size // 2
        flat_a = blocks_a.reshape(-1, bytes_per_row)
        flat_b = blocks_b.reshape(-1, bytes_per_row)
        scale_a = scales_a.reshape(-1, 1)
        scale_b = scales_b.reshape(-1, 1)
    else:
        bytes_per_row = blocks_a.shape[-1]
        flat_a = blocks_a.reshape(-1, bytes_per_row)
        flat_b = blocks_b.reshape(-1, bytes_per_row)
        scale_a = scales_a.reshape(flat_a.shape[0], -1)
        scale_b = scales_b.reshape(flat_b.shape[0], -1)
        rows_per_chunk = max(1, chunk_size // (bytes_per_row * 2))
    lut = torch.tensor(FP4_E2M1_VALUES, dtype=torch.float64)

    def decoded_chunks() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, flat_a.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, flat_a.shape[0])
            outputs: list[torch.Tensor] = []
            for packed, scales, mode in (
                (flat_a[start:end], scale_a[start:end], scale_mode_a),
                (flat_b[start:end], scale_b[start:end], scale_mode_b),
            ):
                packed = (
                    packed.view(torch.uint8)
                    if "float4_e2m1" in str(packed.dtype)
                    else packed.to(torch.uint8)
                )
                indices = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(end - start, -1)
                values = lut[indices.to(torch.long)]
                multipliers = _mxfp_scale(scales, mode)
                if layout_a == "ascend_row":
                    multipliers = multipliers.repeat_interleave(block_size, dim=-1)
                outputs.append(values * multipliers)
            yield outputs[0], outputs[1]

    total = math.prod(shape_a)
    return _compare_chunks(decoded_chunks(), total, sample_size, atol, rtol), shape_a


def load_layer(
    checkpoint: Checkpoint,
    name: str,
    storage_format: str,
    scale_name: str | None,
) -> dict[str, Any]:
    base = name[:-7] if name.endswith(".blocks") else name
    openai_blocks = name if name.endswith(".blocks") else f"{base}.blocks"
    scale_candidates = [
        candidate for candidate in (scale_name, f"{base}.scales", f"{name}_scale") if candidate
    ]
    inferred_scale = next(
        (candidate for candidate in scale_candidates if checkpoint.has_tensor(candidate)),
        scale_candidates[0],
    )
    if checkpoint.has_tensor(openai_blocks):
        blocks_name = openai_blocks
    else:
        # Ascend checkpoints store packed E2M1 directly as `weight`/`w13_weight`
        # and the E8M0 bytes as `weight_scale`/`w13_weight_scale`.
        blocks_name = name
    if storage_format == "auto":
        storage_format = (
            "mxfp4"
            if checkpoint.has_tensor(blocks_name) and checkpoint.has_tensor(inferred_scale)
            else "raw"
        )
    if storage_format == "raw":
        tensor, shard = checkpoint.tensor(name)
        return {"format": "raw", "tensor": tensor, "shard": shard, "name": name}
    if not checkpoint.has_tensor(blocks_name):
        raise KeyError(f"MXFP4 blocks tensor {blocks_name!r} was not found in {checkpoint.path}")
    if not checkpoint.has_tensor(inferred_scale):
        raise KeyError(
            f"MXFP scale tensor {inferred_scale!r} was not found in {checkpoint.path}; "
            "specify it with --scale-layer-a/--scale-layer-b."
        )
    blocks, blocks_shard = checkpoint.tensor(blocks_name)
    scales, scales_shard = checkpoint.tensor(inferred_scale)
    return {
        "format": "mxfp4",
        "blocks": blocks,
        "scales": scales,
        "shard": blocks_shard,
        "scale_shard": scales_shard,
        "name": base,
        "blocks_name": blocks_name,
        "scale_name": inferred_scale,
    }


def _format_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"Tensor A : {report['tensor_a']}",
        f"Tensor B : {report['tensor_b']}",
        f"Shape    : {report['shape']}",
        f"Dtype    : {report['dtype_a']} vs {report['dtype_b']}",
        f"Format   : {report['format_a']} vs {report['format_b']}",
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
    if report["format_a"] == "mxfp4":
        lines += ["", "MXFP4 values were dequantized with their block scales before comparison."]
    elif report["raw_quantized_values"]:
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
    parser.add_argument(
        "--format", choices=("auto", "raw", "mxfp4"), default="auto",
        help="Checkpoint tensor storage format; auto recognizes name.blocks/name.scales",
    )
    parser.add_argument("--scale-layer-a", help="MXFP scale tensor name in model A")
    parser.add_argument("--scale-layer-b", help="MXFP scale tensor name in model B")
    parser.add_argument("--block-size", type=int, default=32, help="MXFP values per scale block")
    parser.add_argument(
        "--scale-mode-a", choices=("auto", "e8m0", "exponent", "float"), default="auto",
        help="Encoding of model A's MXFP scales",
    )
    parser.add_argument(
        "--scale-mode-b", choices=("auto", "e8m0", "exponent", "float"), default="auto",
        help="Encoding of model B's MXFP scales",
    )
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
        if args.block_size <= 0 or args.block_size % 2:
            raise ValueError("--block-size must be a positive even number.")

        input_a = load_layer(checkpoint_a, layer_a, args.format, args.scale_layer_a)
        input_b = load_layer(checkpoint_b, layer_b, args.format, args.scale_layer_b)
        if input_a["format"] != input_b["format"]:
            raise ValueError(
                f"Detected different formats: model A {input_a['format']} vs model B {input_b['format']}. "
                "Use --format or explicit scale tensor names."
            )

        if input_a["format"] == "mxfp4":
            metrics, shape = compare_mxfp4(
                input_a["blocks"], input_a["scales"], input_b["blocks"], input_b["scales"],
                args.block_size, args.scale_mode_a, args.scale_mode_b, args.chunk_size,
                args.sample_size, args.atol, args.rtol,
            )
            dtype_a = f"{input_a['blocks'].dtype} + scale {input_a['scales'].dtype}"
            dtype_b = f"{input_b['blocks'].dtype} + scale {input_b['scales'].dtype}"
            shard_a = f"{input_a['shard']} (scale: {input_a['scale_shard']})"
            shard_b = f"{input_b['shard']} (scale: {input_b['scale_shard']})"
            tensor_name_a = f"{input_a['blocks_name']} + {input_a['scale_name']}"
            tensor_name_b = f"{input_b['blocks_name']} + {input_b['scale_name']}"
            raw_quantized = False
        else:
            tensor_a, tensor_b = input_a["tensor"], input_b["tensor"]
            metrics = compare_tensors(
                tensor_a, tensor_b, args.chunk_size, args.sample_size, args.atol, args.rtol
            )
            shape = tuple(tensor_a.shape)
            dtype_a, dtype_b = str(tensor_a.dtype), str(tensor_b.dtype)
            shard_a, shard_b = str(input_a["shard"]), str(input_b["shard"])
            tensor_name_a, tensor_name_b = layer_a, layer_b
            raw_quantized = not tensor_a.is_floating_point() or not tensor_b.is_floating_point()
        report = {
            "model_a": str(checkpoint_a.path),
            "model_b": str(checkpoint_b.path),
            "tensor_a": tensor_name_a,
            "tensor_b": tensor_name_b,
            "shard_a": str(shard_a),
            "shard_b": str(shard_b),
            "shape": list(shape),
            "dtype_a": dtype_a,
            "dtype_b": dtype_b,
            "format_a": input_a["format"],
            "format_b": input_b["format"],
            "raw_quantized_values": raw_quantized,
            "metrics": metrics,
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
