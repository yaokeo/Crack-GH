from __future__ import annotations

import argparse
import math
from collections import defaultdict

import torch
from torch import Tensor, nn

from crackseg.config import load_config
from crackseg.models import build_model
from crackseg.models.sampling import DySampleCore, HaarDownsample
from crackseg.models.wavelet_akconv import WaveletGuidedAKConv


BILINEAR_SAMPLE_FLOPS = 7  # Four multiplies plus three additions.


def human_count(value: float, unit: str) -> str:
    for scale, prefix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= scale:
            return f"{value / scale:.4f} {prefix}{unit}"
    return f"{value:.0f} {unit}"


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile CrackGH-UNet MACs, parameters, and dynamic sampling FLOPs"
    )
    parser.add_argument("--config", default="configs/wavelet_akconv_crack.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()

    try:
        from thop import profile
    except ImportError as error:
        raise SystemExit("THOP is missing. Install it with: pip install thop") from error

    config = load_config(args.config)
    configured_h, configured_w = config["data"].get("image_size", (512, 512))
    height = int(args.height or configured_h)
    width = int(args.width or configured_w)
    batch_size = int(args.batch_size)
    if min(batch_size, height, width) < 1:
        raise ValueError("batch-size, height, and width must be positive")

    device = select_device(args.device)
    model = build_model(config["model"]).to(device).eval()
    in_channels = int(config["model"].get("in_channels", 3))
    dummy = torch.randn(batch_size, in_channels, height, width, device=device)

    extra_flops: dict[str, float] = defaultdict(float)
    aux_native_shapes: list[tuple[int, int, int, int]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def akconv_hook(
        module: WaveletGuidedAKConv,
        inputs: tuple[Tensor, Tensor | None],
        output: Tensor,
    ) -> None:
        del output
        feature = inputs[0]
        b, c, h, w = feature.shape
        extra_flops["akconv_grid_sample"] += (
            b * c * h * w * module.num_points * BILINEAR_SAMPLE_FLOPS
        )
        # Coordinate normalization and applying the learned 2-D displacement.
        extra_flops["akconv_grid_math"] += b * h * w * module.num_points * 8

    def dysample_hook(
        module: DySampleCore,
        inputs: tuple[Tensor],
        output: Tensor,
    ) -> None:
        del module, inputs
        extra_flops["dysample_grid_sample"] += (
            output.numel() * BILINEAR_SAMPLE_FLOPS
        )

    def haar_hook(
        module: HaarDownsample,
        inputs: tuple[Tensor],
        output: tuple[Tensor, Tensor],
    ) -> None:
        del module, output
        feature = inputs[0]
        b, c, h, w = feature.shape
        out_h = math.ceil(h / 2)
        out_w = math.ceil(w / 2)
        # Each of four Haar bands uses roughly 3 adds and 1 scale multiply.
        extra_flops["haar_decomposition"] += b * c * out_h * out_w * 16

    def aux_hook(
        module: nn.Module,
        inputs: tuple[Tensor],
        output: Tensor,
    ) -> None:
        del module, inputs
        aux_native_shapes.append(tuple(int(value) for value in output.shape))

    for module in model.modules():
        if isinstance(module, WaveletGuidedAKConv):
            handles.append(module.register_forward_hook(akconv_hook))
        elif isinstance(module, DySampleCore):
            handles.append(module.register_forward_hook(dysample_hook))
        elif isinstance(module, HaarDownsample):
            handles.append(module.register_forward_hook(haar_hook))
    for head in getattr(model, "aux_heads", []):
        handles.append(head.register_forward_hook(aux_hook))

    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    finally:
        for handle in handles:
            handle.remove()

    multiple = int(config["model"].get("input_multiple", 8))
    padded_h = math.ceil(height / multiple) * multiple
    padded_w = math.ceil(width / multiple) * multiple
    for b, channels, native_h, native_w in aux_native_shapes:
        if (native_h, native_w) != (padded_h, padded_w):
            extra_flops["aux_bilinear_resize"] += (
                b * channels * padded_h * padded_w * BILINEAR_SAMPLE_FLOPS
            )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    thop_flops = 2.0 * float(macs)
    additional_flops = sum(extra_flops.values())
    estimated_total = thop_flops + additional_flops

    print(f"Input: ({batch_size}, {in_channels}, {height}, {width})")
    print(f"Device: {device}")
    print(f"Parameters: {parameters:,} ({human_count(parameters, 'Params')})")
    print(f"Trainable parameters: {trainable:,}")
    print(f"THOP supported-op MACs: {human_count(float(macs), 'MACs')}")
    print(
        "THOP supported-op FLOPs (1 MAC = 2 FLOPs): "
        f"{human_count(thop_flops, 'FLOPs')}"
    )
    for name, value in sorted(extra_flops.items()):
        print(f"Additional {name}: {human_count(value, 'FLOPs')}")
    print(
        "Estimated total forward FLOPs: "
        f"{human_count(estimated_total, 'FLOPs')}"
    )
    print(
        "Per-image estimated FLOPs: "
        f"{human_count(estimated_total / batch_size, 'FLOPs')}"
    )
    print(
        "Note: the total is an engineering estimate. THOP does not natively count "
        "functional grid_sample/interpolate or every elementwise gate; the script "
        "adds the dominant dynamic-sampling, Haar, and auxiliary-resize costs."
    )


if __name__ == "__main__":
    main()
