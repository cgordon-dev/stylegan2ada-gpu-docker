#!/usr/bin/env python3
import os, glob, re, subprocess, csv
from datetime import datetime
import numpy as np
from PIL import Image

# Config
AEY_MAX   = 100.0
INSTANCE  = 'g4dn.xlarge'
REGION    = os.getenv('AWS_REGION', 'us-east-1')
FID_MIN, FID_MAX = 0.0, 300.0
BASE_TIMES = {
    'train-auto': 25.0, 'train-cifar': 25.0, 'train-mixed': 25.0,
    'generate': 0.5, 'project': 1.0
}

def get_spot_price():
    out = subprocess.check_output([
        'aws','ec2','describe-spot-price-history',
        '--instance-types', INSTANCE,
        '--product-descriptions','Linux/UNIX',
        '--start-time', datetime.utcnow().isoformat()+'Z',
        '--max-items','1',
        '--region', REGION,
        '--query','SpotPriceHistory[0].SpotPrice'
    ])
    return float(out.strip().strip(b'"'))

def parse_fid(txtfile):
    txt = open(txtfile).read()
    m = re.search(r'fid50k_full:\s*([\d.]+)', txt)
    return float(m.group(1)) if m else None

def parse_mse(recon, target):
    a = np.array(Image.open(recon)).astype(float)/255
    b = np.array(Image.open(target)).astype(float)/255
    return np.mean((a - b)**2)

# Gather stats
spot_hr = get_spot_price()
spot_sec = spot_hr/3600.0
results = {}

for stage in ['train-auto','train-cifar','train-mixed','generate','project']:
    # Timing (R)
    t = float(open(f'results/timings/{stage}.sec').read().strip())
    R = t/3600.0

    # Count images processed
    if stage.startswith('train-'):
        kimgs = {'train-auto':200,'train-cifar':100,'train-mixed':100}[stage]
        imgs = kimgs*1000
    elif stage=='generate':
        snaps = sum(len(glob.glob(f'results/{s}/*.pkl')) for s in ['train-auto','train-cifar','train-mixed'])
        imgs = snaps*10
    else:  # project
        snaps = sum(len(glob.glob(f'results/{s}/*.pkl')) for s in ['train-auto','train-cifar','train-mixed'])
        imgs = snaps*1

    # C = $ per 1k images
    C = spot_sec * 1000

    # T = time saved
    unit = BASE_TIMES[stage]
    sec_per_unit = (t/(kimgs if stage.startswith('train-') else imgs))
    T = max(0,1 - sec_per_unit/unit)

    # Q = quality
    if stage in ['train-auto','train-cifar','train-mixed','generate']:
        last = sorted(glob.glob(f'results/metrics/{stage}/*.txt'))[-1]
        fid = parse_fid(last)
        Q = max(0,1 - (fid - FID_MIN)/(FID_MAX - FID_MIN))
    else:
        # project
        recon = sorted(glob.glob('results/proj/train-mixed/*.png'))[-1]
        target = 'datasets/cifar10/airplane/00001.png'
        mse = parse_mse(recon, target)
        Q = max(0,1 - mse)

    AEY_raw  = Q * T / (C * R)
    AEY_norm = AEY_raw / AEY_MAX
    results[stage] = (Q, T, C, R, AEY_raw, AEY_norm)

# Write CSV
with open('results/AEY_report.csv','w') as f:
    w=csv.writer(f)
    w.writerow(['Stage','Q','T','C','$R','AEY_raw','AEY_norm'])
    for s,v in results.items():
        w.writerow([s]+[f"{x:.4g}" for x in v])

print("AEY report → results/AEY_report.csv")