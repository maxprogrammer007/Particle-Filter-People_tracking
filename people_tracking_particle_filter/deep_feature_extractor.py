import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, vgg16, MobileNet_V2_Weights, VGG16_Weights
import numpy as np
import cv2

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Global state
model = None
preprocess = None

def load_model(architecture="mobilenet"):
    global model, preprocess

    if architecture == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights).features.to(device).eval()
        preprocess = weights.transforms()

    elif architecture == "vgg16":
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights).features.to(device).eval()
        preprocess = weights.transforms()

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

@torch.no_grad()
def extract_deep_feature(patch):
    if patch is None or patch.size == 0:
        return torch.zeros(1280 if isinstance(model, torch.nn.Sequential) else 512, device=device)

    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0).to(device)
    fmap = model(tensor)

    # If needed, apply pooling to collapse spatial dimensions
    if len(fmap.shape) == 4:
        pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1)
    else:
        pooled = fmap.view(-1)

    return F.normalize(pooled, dim=0)

@torch.no_grad()
def extract_batch_features(patch_list):
    valid = [p for p in patch_list if p is not None and p.size > 0]
    if not valid:
        return torch.zeros((0, 1280 if isinstance(model, torch.nn.Sequential) else 512), device=device)

    batch = torch.stack([
        preprocess(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in valid
    ]).to(device)

    fmap = model(batch)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).squeeze(-1).squeeze(-1)
    return F.normalize(pooled, p=2, dim=1)
