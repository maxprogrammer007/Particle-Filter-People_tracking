import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from config import FEATURE_EXTRACTOR_ARCH
from torchvision.models import (
    mobilenet_v2, vgg16, densenet121,
    MobileNet_V2_Weights, VGG16_Weights, DenseNet121_Weights
)
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Global state
model = None
preprocess = None
feature_dim = None

def load_model(architecture: str = FEATURE_EXTRACTOR_ARCH):
    """Load and initialize the chosen backbone + transforms."""
    global model, preprocess, feature_dim

    if architecture.lower() == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights).features.to(device).eval()
        preprocess = weights.transforms()
        feature_dim = 1280

    elif architecture.lower() == "vgg16":
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights).features.to(device).eval()
        preprocess = weights.transforms()
        feature_dim = 512
        
    elif architecture.lower() == "densenet121":
        weights = DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights).features.to(device).eval()
        preprocess = weights.transforms()
        feature_dim = 1024  # DenseNet121 has 1024 output features
        
    elif architecture.lower() == "efficientnet":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights).features.to(device).eval()
        preprocess = weights.transforms()
        feature_dim = 1280


    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    print(f"[INFO] Loaded feature extractor: {architecture} (dim={feature_dim})")

# Auto‑load at import time
load_model()

@torch.no_grad()
def extract_deep_feature(patch):
    """Extract a single normalized feature vector from an OpenCV BGR patch."""
    global model, preprocess, feature_dim

    if model is None or preprocess is None:
        # Fallback if something went wrong
        load_model()

    if patch is None or patch.size == 0:
        return torch.zeros(feature_dim, device=device)

    # Convert BGR→RGB→PIL
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Preprocess + forward
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    fmap = model(tensor)

    # Pool to vector
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1)
    return F.normalize(pooled, dim=0)

@torch.no_grad()
def extract_batch_features(patch_list):
    """Batch‐extract features for a list of OpenCV patches."""
    global model, preprocess, feature_dim

    if model is None or preprocess is None:
        load_model()

    # Keep only valid patches
    valid = [p for p in patch_list if p is not None and p.size > 0]
    if not valid:
        return torch.zeros((0, feature_dim), device=device)

    # Convert each to PIL and preprocess
    tensors = []
    for p in valid:
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(preprocess(pil))

    batch = torch.stack(tensors).to(device)
    fmap = model(batch)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).squeeze(-1).squeeze(-1)
    return F.normalize(pooled, p=2, dim=1)
