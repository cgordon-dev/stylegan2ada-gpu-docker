#!/usr/bin/env python3
import os
import tarfile
from datetime import datetime
import boto3

BUCKET = 'aws-gpu-monitoring-logs'
STAGES = [
    ('train-auto',  'Training baseline'),
    ('train-cifar', 'Fine‑tune CIFAR‑10'),
    ('train-mixed','Mixed‑precision'),
    ('generated',   'Generated samples'),
    ('proj',        'Latent projections'),
    ('metrics',     'Computed metrics'),
    ('timings',     'Stage durations'),
]

def make_archive(source_dir, archive_path):
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def upload_file(s3_client, file_path, s3_key, tags):
    s3_client.upload_file(
        Filename=file_path,
        Bucket=BUCKET,
        Key=s3_key,
        ExtraArgs={'Tagging': '&'.join(f'{k}={v}' for k,v in tags.items())}
    )
    print(f"Uploaded {file_path} → s3://{BUCKET}/{s3_key}")

def main():
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))

    for stage, desc in STAGES:
        src = f"/workspace/results/{stage}"
        if not os.path.exists(src):
            print(f"Skipping {stage}, nothing to upload.")
            continue

        archive_name = f"{stage}-{timestamp}.tar.gz"
        archive_path = os.path.join("/workspace/results", archive_name)
        print(f"\nPackaging {desc} → {archive_name}")
        make_archive(src, archive_path)

        s3_key = f"{stage}/{timestamp}/{archive_name}"
        tags = {'stage': stage, 'run_timestamp': timestamp}
        upload_file(s3, archive_path, s3_key, tags)

if __name__ == "__main__":
    main()