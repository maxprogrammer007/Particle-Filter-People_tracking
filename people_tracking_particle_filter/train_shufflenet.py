# train_shufflenet.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from PIL import Image
from multiprocessing import freeze_support

# --- Configuration ---
DATA_DIR    = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\dataset\\patches"        # adjust if different
NUM_EPOCHS  = 10
BATCH_SIZE  = 32
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = "shufflenet_finetuned.pth"

# --- Custom Dataset ---
class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        self.samples = []
        # Collect (path, class) tuples
        for fname in os.listdir(root_dir):
            if not fname.lower().endswith((".jpg",".png")):
                continue
            parts = fname.rsplit("_", 1)[-1]  # e.g. "person3.jpg"
            cls = int(parts.replace("person","").split(".")[0])
            self.samples.append((os.path.join(root_dir, fname), cls))
        self.classes = sorted({cls for _, cls in self.samples})
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = self.class_to_idx[cls]
        return x, y

def main():
    # Ensure safe multiprocessing start on Windows
    freeze_support()

    # --- Model Setup ---
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    model = shufflenet_v2_x1_0(weights=weights)
    # Replace classifier head
    dataset = PatchDataset(DATA_DIR)
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    # --- Training Tools ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- DataLoader ---
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Training Loop ---
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        correct = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
        avg_loss = total_loss / len(dataset)
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {avg_loss:.4f}  Acc: {acc:.4f}")

    # --- Save Checkpoint ---
    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"[INFO] Saved fine-tuned model to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
