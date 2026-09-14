from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from crackseg.config import load_config
from crackseg.models import build_model


def image_to_tensor(image: Image.Image, in_channels: int, normalize: str) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if normalize == "imagenet" and in_channels == 3:
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    elif normalize == "half":
        tensor = (tensor - 0.5) / 0.5
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="CrackGH-UNet single-image inference")
    parser.add_argument("--config", default="configs/base_crack.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    data_config = config["data"]
    in_channels = int(config["model"].get("in_channels", 3))
    mode = "RGB" if in_channels == 3 else "L"
    with Image.open(args.image) as source:
        image = source.convert(mode)
        original_size = image.size
        target_h, target_w = data_config.get("image_size", (512, 512))
        resized = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    tensor = image_to_tensor(
        resized,
        in_channels,
        data_config.get("normalize", "imagenet"),
    ).to(device)

    with torch.inference_mode():
        logits = model(tensor)["logits"]
        probability = torch.sigmoid(logits)
        probability = F.interpolate(
            probability,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    mask = (probability >= args.threshold).to(torch.uint8).cpu().numpy() * 255
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)
    print(f"Saved mask to {output_path}")


if __name__ == "__main__":
    main()
