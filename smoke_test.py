from __future__ import annotations

import torch

from crackseg.losses import CompositeCrackLoss
from crackseg.models import CrackGHUNet


def main() -> None:
    torch.manual_seed(7)
    model = CrackGHUNet(
        widths=(16, 24, 32, 48),
        encoder_groups=(8, 4, 2),
        decoder_groups=(4, 8, 2),
    )
    image = torch.randn(2, 3, 127, 131)
    target = (torch.rand(2, 1, 127, 131) > 0.92).float()
    output = model(image)
    criterion = CompositeCrackLoss(cldice_weight=0.1, skeleton_iterations=3)
    loss = criterion(output, target)
    loss.backward()
    print(f"main logits: {tuple(output['logits'].shape)}")
    print(f"aux logits: {[tuple(item.shape) for item in output['aux_logits']]}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"loss: {loss.item():.6f}")
    print("smoke test: OK")


if __name__ == "__main__":
    main()
