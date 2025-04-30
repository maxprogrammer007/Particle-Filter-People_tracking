import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision import transforms
import os

# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to optional fine-tuned model
FINETUNED_PATH = "mobilenetv2_finetuned.pth"  # (you can rename this to shufflenet_finetuned.pth if applicable)

# Load ShuffleNetV2
weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
preprocess = weights.transforms()

cclass FeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, finetuned_path=None, device_type="cuda"):
        super().__init__()
        base = shufflenet_v2_x1_0(weights=weights if pretrained else None)
        self.backbone = torch.nn.Sequential(
            base.conv1,
            base.maxpool,
            base.stage2,
            base.stage3,
            base.stage4,
            base.conv5
        )
        if finetuned_path and os.path.exists(finetuned_path):
            print("[INFO] Loading fine-tuned ShuffleNetV2 weights...")
            state_dict = torch.load(finetuned_path, map_location=device)
            base.load_state_dict(state_dict, strict=False)

        self.device_type = device_type
        self.backbone.to(device).eval()

    def forward(self, x):
        with torch.amp.autocast(device_type=self.device_type):
            return self.backbone(x)


    
# TorchScript Compilation for performance
model = FeatureExtractor(pretrained=True, finetuned_path=None)
model.eval()
model = torch.jit.script(model)

@torch.no_grad()
@torch.cuda.amp.autocast()
def extract_deep_feature(patch):
    if patch is None or patch.size == 0:
        return torch.zeros(1024, device=device)

    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(device)

    fmap = model(tensor)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1)
    return F.normalize(pooled, dim=0)

@torch.no_grad()
@torch.cuda.amp.autocast()
def extract_batch_features(patch_list):
    valid = [p for p in patch_list if p is not None and p.size > 0]
    if not valid:
        return torch.zeros((0, 1024), device=device)

    batch = []
    for patch in valid:
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(patch_rgb)
        tensor = preprocess(pil_img)
        batch.append(tensor)

    batch_tensor = torch.stack(batch).to(device)
    fmap = model(batch_tensor)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).squeeze(-1).squeeze(-1)
    return F.normalize(pooled, p=2, dim=1)
