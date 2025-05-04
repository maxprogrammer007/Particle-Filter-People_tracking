import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision import transforms
import os

# ------------------ Config ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNED_PATH = "shufflenet_finetuned.pth"  # Replace if you have a fine-tuned checkpoint
FP16 = True  # Set to False if you want full precision

# ------------------ Load Model ------------------
weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
preprocess = weights.transforms()

# Build feature backbone
_base = shufflenet_v2_x1_0(weights=weights)
_model = torch.nn.Sequential(
    _base.conv1,
    _base.maxpool,
    _base.stage2,
    _base.stage3,
    _base.stage4,
    _base.conv5
)
# Load fine-tuned weights if available
if os.path.exists(FINETUNED_PATH):
    print("[INFO] Loading fine-tuned weights...")
    state_dict = torch.load(FINETUNED_PATH, map_location=device)
    _model.load_state_dict(state_dict, strict=False)

_model = _model.to(device).eval()

@torch.no_grad()
def extract_features(patch_list):
    """
    Given a list of OpenCV BGR patches, returns a tensor of shape [N, 1024],
    where N = number of valid patches. Uses FP16 if enabled.
    """
    # Filter out empty patches
    valid_patches = [p for p in patch_list if p is not None and p.size != 0]
    if not valid_patches:
        return torch.zeros((0, 1024), device=device)

    # Preprocess all crops into a single batched tensor
    tensors = []
    for patch in valid_patches:
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t = preprocess(pil)  # [C,H,W]
        tensors.append(t)
    batch = torch.stack(tensors, dim=0).to(device)  # [N,C,H,W]

    # Forward pass in FP16 or FP32
    with torch.amp.autocast(device_type="cuda", enabled=FP16):
        fmap = _model(batch)  # [N, feature_map_channels, h, w]

    # Global average pool to [N, C]
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(fmap.size(0), -1)
    # L2-normalize along the channel dimension
    feats = F.normalize(pooled, p=2, dim=1)
    return feats  # [N, 1024]
