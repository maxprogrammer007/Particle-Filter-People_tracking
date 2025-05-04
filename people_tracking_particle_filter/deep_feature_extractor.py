import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision import transforms
import os

# ------------------ Config ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNED_PATH = r"C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\shufflenet_finetuned.pth"          # if you have fine-tuned weights
SCRIPTED_PATH  = r"C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\shufflenet_scripted.pt"
FP16 = True  # half-precision toggle

# ------------------ Load Model ------------------
weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
preprocess = weights.transforms()

if os.path.exists(SCRIPTED_PATH):
    print(f"[INFO] Loading scripted ShuffleNet from {SCRIPTED_PATH}")
    _model = torch.jit.load(SCRIPTED_PATH).to(device).eval()
else:
    print("[INFO] Building Python ShuffleNet backbone...")
    base = shufflenet_v2_x1_0(weights=weights)
    _model = torch.nn.Sequential(
        base.conv1,
        base.maxpool,
        base.stage2,
        base.stage3,
        base.stage4,
        base.conv5
    )
    if os.path.exists(FINETUNED_PATH):
        print("[INFO] Loading fine-tuned weights...")
        state_dict = torch.load(FINETUNED_PATH, map_location=device)
        _model.load_state_dict(state_dict, strict=False)
    _model = _model.to(device).eval()

@torch.no_grad()
def extract_features(patch_list):
    """
    Given a list of OpenCV BGR patches, returns a tensor of shape [N, 1024].
    Uses FP16 if enabled and the (scripted) ShuffleNet backbone.
    """
    # Filter out invalid patches
    valid = [p for p in patch_list if p is not None and p.size != 0]
    if not valid:
        return torch.zeros((0, 1024), device=device)

    # Preprocess and batch
    tensors = []
    for patch in valid:
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(preprocess(pil))
    batch = torch.stack(tensors, dim=0).to(device)  # [N,C,H,W]

    # Forward through scripted/model
    with torch.amp.autocast(device_type="cuda", enabled=FP16):
        fmap = _model(batch)  # [N, Cf, Hf, Wf]

    # Global avg pool to [N, Cf]
    pooled = F.adaptive_avg_pool2d(fmap, (1, 1)).view(fmap.size(0), -1)
    # L2 normalize
    feats = F.normalize(pooled, p=2, dim=1)
    return feats  # [N, 1024]
