import torch
import torch.nn.functional as F
import cv2
from PIL import Image  # ✅ NEW
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Global device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 and preprocessing
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).features.to(device).eval()
preprocess = weights.transforms()

@torch.no_grad()
def extract_deep_feature(patch):
    """
    Extract normalized 1280-d feature vector from a single image patch.
    """
    if patch is None or patch.size == 0:
        return torch.zeros(1280, device=device)

    # ✅ Convert OpenCV BGR to RGB → PIL Image
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)

    tensor = preprocess(pil_img).unsqueeze(0).to(device)

    fmap = model(tensor)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1)
    return F.normalize(pooled, dim=0)

@torch.no_grad()
def extract_batch_features(patch_list):
    """
    Batch extract for a list of patches.
    """
    from PIL import Image
    valid = [p for p in patch_list if p is not None and p.size > 0]
    if not valid:
        return torch.zeros((0, 1280), device=device)

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
