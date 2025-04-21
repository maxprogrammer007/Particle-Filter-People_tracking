import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import cv2

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 backbone with pretrained weights
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).features.to(device).eval()

# ImageNet mean/std normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=weights.meta['mean'], std=weights.meta['std'])
])

@torch.no_grad()
def extract_deep_feature(patch):
    """
    Extracts feature vector from a single patch.
    """
    patch_tensor = transform(patch).unsqueeze(0).to(device)
    feature_map = model(patch_tensor)
    feature_vector = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze().cpu().numpy()
    return feature_vector / np.linalg.norm(feature_vector)  # Normalize

@torch.no_grad()
def extract_batch_features(patch_list):
    """
    Batch process multiple patches and return their normalized feature vectors.
    Input: list of cv2 patches
    Output: list of numpy feature vectors
    """
    batch = []
    for patch in patch_list:
        patch_tensor = transform(patch)
        batch.append(patch_tensor)

    batch_tensor = torch.stack(batch).to(device)
    features = model(batch_tensor)
    pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
    normalized = F.normalize(pooled, p=2, dim=1)
    return normalized.cpu().numpy()
