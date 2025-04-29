import torch
import shap
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import torch.nn as nn

# ---------------- Settings ----------------
FRAME_FOLDER = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\people_tracking_particle_filter\\sample_frames"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PLOTS = True
MODEL_PATH = "mobilenetv2_finetuned.pth"

# ---------------- Utility ----------------
def remove_inplace(module):
    for child_name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.Hardswish, nn.Hardtanh)):
            setattr(module, child_name, type(child)(inplace=False))
        else:
            remove_inplace(child)

# ---------------- Load Fine-Tuned Model ----------------
print("[INFO] Loading Fine-Tuned MobileNetV2...")

assert os.path.exists(MODEL_PATH), f"❌ Fine-tuned model not found at {MODEL_PATH}"

model = mobilenet_v2(weights=None)
remove_inplace(model)
model.classifier[1] = nn.Linear(model.last_channel, 1)  # Match fine-tuned output
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------- Load Frames ----------------
def load_frames(folder_path, max_frames=8):
    frames = []
    for filename in sorted(os.listdir(folder_path))[:max_frames]:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        frames.append(preprocess(img))
    return torch.stack(frames)

inputs = load_frames(FRAME_FOLDER).to(DEVICE)
background = inputs[:2]
test_samples = inputs[2:6]

# ---------------- SHAP Analysis ----------------
print("[INFO] Setting up SHAP GradientExplainer...")

explainer = shap.GradientExplainer(model, background)

print("[INFO] Computing SHAP values...")
test_samples.requires_grad = True  # 👈 Enable gradient tracking

shap_values, _ = explainer.shap_values(test_samples, ranked_outputs=1)


print(f"[DEBUG] shap_values original shape: {shap_values.shape}")

# Handle possible batch size squeeze issue
if shap_values.shape[-1] == 1:
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
    np.array([img.detach().numpy().transpose(1, 2, 0) for img in test_samples_denorm]),
    0.0, 1.0
)

shap.image_plot(shap_values, images_to_plot)

if SAVE_PLOTS:
    print("[INFO] Saving SHAP plot as 'shap_output.png'...")
    plt.savefig("shap_output.png")
else:
    plt.show()

print("[INFO] SHAP Analysis Completed ✅")
