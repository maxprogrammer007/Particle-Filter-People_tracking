import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
import os

# Global device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if fine-tuned model exists
FINETUNED_PATH = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\mobilenetv2_finetuned.pth"

# Preprocessing
weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()

# Define scalar-output model if needed
class ScalarMobileNetV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v2
        self.features = mobilenet_v2(weights=None).features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.scalar = torch.nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        return self.pool(x)  # ✅ Return features for consistency

# Load the correct model
if os.path.exists(FINETUNED_PATH):
    print("[INFO] Loading Fine-Tuned MobileNetV2 for feature extraction...")
    model = ScalarMobileNetV2().to(device)
    state_dict = torch.load(FINETUNED_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.features.eval()
else:
    print("[INFO] Using default MobileNetV2 features (ImageNet pretrained)")
    model = mobilenet_v2(weights=weights).features.to(device).eval()

@torch.no_grad()
def extract_deep_feature(patch):
    """
    Extract normalized 1280-d feature vector from a single image patch.
    """
    if patch is None or patch.size == 0:
        return torch.zeros(1280, device=device)

    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(patch_rgb)
    tensor = preprocess(pil_img).unsqueeze(0).to(device)

    fmap = model(tensor)
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1)
    return F.normalize(pooled, dim=0)

@torch.no_grad()
def extract_batch_features(patch_list):
    """
    Batch extract features for a list of patches.
    """
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
