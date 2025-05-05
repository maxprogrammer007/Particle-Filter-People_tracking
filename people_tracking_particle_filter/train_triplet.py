# train_triplet.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from PIL import Image
import random

# --- Configuration ---
PATCH_DIR   = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\dataset\\patches"
NUM_EPOCHS  = 5
BATCH_SIZE  = 32
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = "shufflenet_triplet.pt"

# --- Triplet Dataset ---
class TripletPatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        # Group patches by person ID
        by_person = {}
        for fname in os.listdir(root_dir):
            if not fname.lower().endswith((".jpg",".png")):
                continue
            pid = int(fname.rsplit("_",1)[-1].split(".")[0].replace("person",""))
            by_person.setdefault(pid, []).append(os.path.join(root_dir, fname))
        # Keep only persons with at least 2 samples
        self.by_person = {pid: paths for pid, paths in by_person.items() if len(paths) >= 2}
        if not self.by_person:
            raise ValueError(f"No person has ≥2 patches in {root_dir}")
        self.person_ids = list(self.by_person.keys())

    def __len__(self):
        # arbitrary large length to sample many triplets
        return 100000

    def __getitem__(self, idx):
        # Sample anchor/positive from same person
        pid = random.choice(self.person_ids)
        pos_paths = random.sample(self.by_person[pid], 2)
        # Sample negative from different person
        neg_pid = random.choice([p for p in self.person_ids if p != pid])
        neg_path = random.choice(self.by_person[neg_pid])

        def load_image(path):
            img = Image.open(path).convert("RGB")
            return self.transform(img)

        anchor   = load_image(pos_paths[0])
        positive = load_image(pos_paths[1])
        negative = load_image(neg_path)
        return anchor, positive, negative

def main():
    # Dataset and loader
    dataset = TripletPatchDataset(PATCH_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Backbone network
    base = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
    backbone = nn.Sequential(
        base.conv1, base.maxpool,
        base.stage2, base.stage3,
        base.stage4, base.conv5
    ).to(DEVICE)

    optimizer = torch.optim.Adam(backbone.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    # Training loop
    backbone.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            fa = backbone(anchor).mean(dim=[2,3])
            fp = backbone(positive).mean(dim=[2,3])
            fn = backbone(negative).mean(dim=[2,3])
            loss = loss_fn(fa, fp, fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(loader):.4f}")

    # Script and save
    scripted = torch.jit.script(backbone)
    scripted.save(OUTPUT_PATH)
    print(f"[INFO] Saved triplet-model to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
