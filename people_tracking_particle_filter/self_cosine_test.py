# self_cosine_test.py

import cv2
from deep_feature_extractor import extract_features
import numpy as np

# load two copies of the same patch
patch_path = "C:\\Users\\abhin\\OneDrive\\Documents\\GitHub\\Particle-Filter-People_tracking\\dataset\\patches\\frame0_person0.jpg"
img = cv2.imread(patch_path)
assert img is not None, "Patch not found"

# resize to 224×224 to mimic pipeline
img224 = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)

# extract feature twice
f1 = extract_features([img224])[0].cpu().numpy()
f2 = extract_features([img224])[0].cpu().numpy()

# cosine similarity
cos = np.dot(f1, f2) / (np.linalg.norm(f1)*np.linalg.norm(f2))
print(f"Self‐cosine similarity: {cos:.4f}")
