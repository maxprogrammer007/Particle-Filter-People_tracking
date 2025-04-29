# blob_patch_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class PatchDataset(Dataset):
    def __init__(self, root_dir="C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\dataset\\patches", transform=None):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in sorted(os.listdir(root_dir))
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)
