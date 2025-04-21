import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Select device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 backbone and its recommended preprocessing
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).features.to(device).eval()
preprocess = weights.transforms()  # includes Resize, ToTensor, Normalize

@torch.no_grad()
def extract_deep_feature(patch, device_arg=None):
    """
    Extracts a normalized 1280-d feature vector from a single image patch.
    Automatically runs on GPU if available.
    """
    if patch is None or patch.size == 0:
        return torch.zeros(1280, device=device)

    # Ensure model is on the correct device (ignore device_arg, use global device)
    global model
    if next(model.parameters()).device != device:
        model.to(device)

    # Convert BGR→RGB, apply preprocess, add batch dim
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0).to(device)

    fmap = model(tensor)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1)
    return F.normalize(pooled, dim=0)

@torch.no_grad()
def extract_batch_features(patch_list, device_arg=None):
    """
    Batch extract for a list of patches. Returns
    a torch.Tensor of shape (N,1280), normalized.
    """
    valid = [p for p in patch_list if p is not None and p.size > 0]
    if not valid:
        return torch.zeros((0, 1280), device=device)

    global model
    if next(model.parameters()).device != device:
        model.to(device)

    # Preprocess all patches
    batch = torch.stack([
        preprocess(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
        for p in valid
    ]).to(device)

    fmap = model(batch)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).squeeze(-1).squeeze(-1)
    return F.normalize(pooled, p=2, dim=1)
