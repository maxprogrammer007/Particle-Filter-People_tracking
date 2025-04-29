# fine_tune_mobilenet.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torchvision.transforms as T
import torch.optim as optim
from tqdm import tqdm
from blob_patch_dataset import PatchDataset

# ------------------- Configuration -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\dataset\\patches"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 1  # Binary classification (Person = 1, No-Person = 0)

# ------------------- Dataset -------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Positive-only dataset (assumes patches are human)
class PositiveOnlyDataset(PatchDataset):
    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        y = torch.tensor(1.0)  # Always person
        return x, y

train_dataset = PositiveOnlyDataset(root_dir=ROOT_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------- Model -------------------
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# Patch classifier for binary output
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ------------------- Training -------------------
criterion = nn.BCEWithLogitsLoss()  # Use sigmoid + BCE
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("[INFO] Starting Fine-Tuning...")
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

# ------------------- Save -------------------
save_path = "mobilenetv2_finetuned.pth"
torch.save(model.state_dict(), save_path)
print(f"[INFO] Fine-tuned MobileNetV2 saved as '{save_path}'")
