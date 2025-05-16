#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pickle

# Paths
raw = 'datasets/cifar10_raw/cifar-10-batches-py'
out = 'cifar10_images'
os.makedirs(out, exist_ok=True)

def load_batch(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = d[b'data']            # shape (10000, 3072)
    labels = d[b'labels']
    return data, labels

idx = 0
for batch in sorted(os.listdir(raw)):
    if not batch.startswith('data_batch'): continue
    data, labels = load_batch(os.path.join(raw, batch))
    # reshape to (N,3,32,32)
    data = data.reshape(-1,3,32,32)
    for img_arr in tqdm(data, desc=f"Batch {batch}"):
        # move channels to last dim
        img = np.transpose(img_arr, (1,2,0))
        # save as PNG
        Image.fromarray(img).save(f"{out}/{idx:05d}.png")
        idx += 1

print("Saved", idx, "images to", out)