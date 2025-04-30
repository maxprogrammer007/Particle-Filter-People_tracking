import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision import transforms
import os

# ------------------ Config ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNED_PATH = "shufflenet_finetuned.pth"  # Replace if fine-tuned

# ------------------ Load Model ------------------
weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
preprocess = weights.transforms()

class FeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, finetuned_path=None):
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
            print("[INFO] Loading fine-tuned weights...")
            state_dict = torch.load(finetuned_path, map_location=device)
            base.load_state_dict(state_dict, strict=False)
        self.backbone.to(device).eval()

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda"):
            return self.backbone(x)

# Optional TorchScript (comment out if debugging)
model = FeatureExtractor(pretrained=True, finetuned_path=None)
model.eval()
model = torch.jit.script(model)

# ------------------ Feature Extraction ------------------
@torch.no_grad()
def extract_deep_feature(patch):
    if patch is None or patch.size == 0:
        return torch.zeros(1024, device=device)

    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type="cuda"):
        fmap = model(tensor)

    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1)
    return F.normalize(pooled, dim=0)

@torch.no_grad()
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

    with torch.amp.autocast(device_type="cuda"):
        fmap = model(batch_tensor)

    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).squeeze(-1).squeeze(-1)
    return F.normalize(pooled, p=2, dim=1)
