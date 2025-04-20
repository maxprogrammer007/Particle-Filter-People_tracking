import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

# Load pretrained ResNet18 model and remove the classifier
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet means
                         std=[0.229, 0.224, 0.225])
])

def extract_deep_feature(patch: np.ndarray) -> torch.Tensor:
    """
    Extract a 512-dim deep feature vector from an image patch using ResNet18.
    :param patch: BGR OpenCV image patch (numpy array)
    :return: 512-dimensional PyTorch tensor
    """
    if patch.size == 0:
        return torch.zeros(512)

    # Convert to RGB and preprocess
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(patch_rgb).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        feature = model(input_tensor)

    return feature.view(-1)  # Flatten to shape [512]
