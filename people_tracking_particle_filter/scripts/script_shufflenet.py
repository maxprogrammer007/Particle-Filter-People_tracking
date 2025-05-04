# scripts/script_shufflenet.py

import torch
import os
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

def main():
    # 1) Instantiate the same Sequential backbone you use in deep_feature_extractor
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    base = shufflenet_v2_x1_0(weights=weights)
    backbone = torch.nn.Sequential(
        base.conv1,
        base.maxpool,
        base.stage2,
        base.stage3,
        base.stage4,
        base.conv5
    )
    backbone.eval()

    # 2) Script it
    scripted = torch.jit.script(backbone)

    # 3) Ensure script output directory exists (project root)
    output_path = os.path.join(os.path.dirname(__file__), "..", "shufflenet_scripted.pt")
    output_path = os.path.abspath(output_path)

    # 4) Save to disk
    scripted.save(output_path)
    print(f"[INFO] Saved scripted model to {output_path}")

if __name__ == "__main__":
    main()
