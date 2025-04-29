import torch
import shap
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

# ---------------- Settings ----------------
FRAME_FOLDER = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\sample_frames"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PLOTS = True

# ---------------- Utility ----------------
def remove_inplace(module):
    for child_name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.Hardswish, nn.Hardtanh)):
            setattr(module, child_name, type(child)(inplace=False))
        else:
            remove_inplace(child)

# ---------------- Load Model ----------------
print("[INFO] Loading MobileNetV2...")
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
remove_inplace(model)
model = model.to(DEVICE).eval()

preprocess = weights.transforms()

# ---------------- Load Frames ----------------
def load_frames(folder_path, max_frames=8):
    frames = []
    for filename in sorted(os.listdir(folder_path))[:max_frames]:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        frames.append(preprocess(img))
    return torch.stack(frames)

inputs = load_frames(FRAME_FOLDER).to(DEVICE)

# ---------------- SHAP Analysis ----------------
print("[INFO] Setting up SHAP GradientExplainer...")
background = inputs[:2]
test_samples = inputs[2:6]

explainer = shap.GradientExplainer(model, background)

print("[INFO] Computing SHAP values...")
shap_values, indexes = explainer.shap_values(test_samples, ranked_outputs=1)

print(f"[DEBUG] shap_values original shape: {shap_values.shape}")

# Remove extra 1-size dimension
shap_values = np.squeeze(shap_values, axis=-1)
print(f"[DEBUG] shap_values after squeeze: {shap_values.shape}")

# Rearrange axes (C, H, W) -> (H, W, C)
shap_values = np.transpose(shap_values, (0, 2, 3, 1))
print(f"[DEBUG] shap_values after transpose: {shap_values.shape}")

# ---------------- Visualization ----------------
print("[INFO] Plotting SHAP results...")

# Denormalize test inputs
inv_normalize = T.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)
test_samples_denorm = torch.stack([inv_normalize(x.cpu()) for x in test_samples])

# Convert to (H, W, C)
images_to_plot = np.clip(
    np.array([img.numpy().transpose(1, 2, 0) for img in test_samples_denorm]),
    0.0, 1.0
)

# SHAP plot
shap.image_plot(shap_values, images_to_plot)

# Save or show
if SAVE_PLOTS:
    print("[INFO] Saving SHAP plot as 'shap_output.png'...")
    plt.savefig("shap_output.png")
else:
    plt.show()

# ---------------- Layer-wise SHAP Importance ----------------
print("[INFO] Computing per-layer SHAP summaries...")
layer_activations = {}

def register_hook(layer_name):
    def hook_fn(module, input, output):
        layer_activations[layer_name] = output.detach().cpu().numpy()
    return hook_fn

for name, module in model.features.named_children():
    module.register_forward_hook(register_hook(f"features.{name}"))

# Run a forward pass to collect activations
_ = model(test_samples)

# Print L2 norm summary of layer activations
print("\n[Layer SHAP Activation Summary]")
for name, act in layer_activations.items():
    norm = np.linalg.norm(act)
    print(f"{name}: L2 norm = {norm:.4f}")

print("[INFO] SHAP Analysis Completed ✅")
