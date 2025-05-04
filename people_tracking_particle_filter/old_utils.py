# old_utils.py

import torch
import torchvision.transforms as T
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from PIL import Image

# Singleton feature‐extraction network + its device
_feature_model = None
_feature_device = None

# Standard ImageNet preprocessing
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

def _get_feature_model(device: torch.device):
    global _feature_model, _feature_device
    if _feature_model is None or _feature_device != device:
        # new API: explicitly pick DEFAULT for the latest pretrained weights
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
        net = shufflenet_v2_x1_0(weights=weights)
        # drop the final FC, keep conv layers + global avg
        modules = list(net.children())[:-1]
        modules.append(torch.nn.AdaptiveAvgPool2d(1))
        model = torch.nn.Sequential(*modules).to(device).eval()
        _feature_model = model
        _feature_device = device
    return _feature_model
# old_utils.py
@torch.no_grad()
def extract_batch_features(img_list, device: torch.device):
    """
    GPU‐enabled ShuffleNetV2 features.

    Args:
      img_list: list of H×W×3 uint8 BGR numpy patches
      device:   torch.device("cuda") or "cpu"

    Returns:
      Tensor of shape (N,1024) on `device`.
    """
    if len(img_list) == 0:
        # nothing to do
        return torch.empty(0, 1024, device=device)

    model = _get_feature_model(device)
    tensors = []
    for img in img_list:
        rgb = img[..., ::-1]
        pil = Image.fromarray(rgb)
        t   = _transform(pil)
        tensors.append(t.unsqueeze(0))

    # you can optionally log here if you really want
    # print(f"extracting features for {len(tensors)} patches")

    batch = torch.cat(tensors, dim=0).to(device)
    feats = model(batch)
    return feats.view(feats.size(0), -1)
