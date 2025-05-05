# train_triplet.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from PIL import Image
import random

class TripletPatchDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        # Load all patches per person
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
        self.by_person = {}
        for f in os.listdir(patch_dir):
            if not f.endswith(('.jpg','.png')): continue
            pid = int(f.rsplit('_',1)[-1].split('.')[0].replace('person',''))
            self.by_person.setdefault(pid,[]).append(os.path.join(patch_dir,f))
        self.person_ids = list(self.by_person.keys())

    def __len__(self): return 100000  # sample size
    def __getitem__(self, _):
        # Sample anchor person
        pid = random.choice(self.person_ids)
        pos = random.sample(self.by_person[pid],2)
        neg_pid = random.choice([p for p in self.person_ids if p!=pid])
        neg  = random.choice(self.by_person[neg_pid])

        def load(path):
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        return load(pos[0]), load(pos[1]), load(neg)

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ds = TripletPatchDataset('dataset/patches')
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

    # Backbone
    base = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
    backbone = nn.Sequential(base.conv1, base.maxpool, base.stage2,
                             base.stage3, base.stage4, base.conv5).to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    backbone.train()
    for epoch in range(5):
        total=0
        for a,p,n in loader:
            a,p,n = a.to(device), p.to(device), n.to(device)
            fa, fp, fn = backbone(a), backbone(p), backbone(n)
            la = loss_fn(fa.mean([2,3]), fp.mean([2,3]), fn.mean([2,3]))
            optimizer.zero_grad(); la.backward(); optimizer.step()
            total+=la.item()
        print(f'Epoch {epoch+1} loss {total/len(loader):.4f}')
    torch.jit.script(backbone).save('shufflenet_triplet.pt')
