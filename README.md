# StyleGAN2‑ADA Dockerized Monorepo

A one‑stop repository to train, fine‑tune, and generate images with NVIDIA’s StyleGAN2‑ADA in PyTorch **and** monitor your workloads with Prometheus, Grafana, cAdvisor, and NVIDIA DCGM Exporter. This monorepo bundles:

- **stylegan2-ada-pytorch/**: Official GAN code, Dockerfile, and multi‑workload `docker-compose.yml` (standard training, CIFAR‑10 fine‑tuning, mixed‑precision, inference, vectorized inference).
- **monitoring/**: Prometheus + Grafana + Node Exporter + cAdvisor + DCGM Exporter stack with preconfigured dashboards and scrape configs.
- **datasets/**: Place your `cifar10.zip` (or other dataset zips) here.
- **outputs/**: Mounted output folders for each workload.
- **upload_all_to_s3.sh**: Fully automated script to sync logs, models, and metrics to S3, tagging with EC2 Instance ID and IP.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
1. [Directory Layout](#directory-layout)
1. [Setup on EC2](#setup-on-ec2)
1. [Workload Execution](#workload-execution)
    - [Prepare Dataset](#prepare-dataset)
    - [Start Workloads](#start-workloads)
    - [Upload to S3](#upload-to-s3)
1. [Monitoring Stack](#monitoring-stack)
1. [Development & Contributing](#development--contributing)

---

## Prerequisites

- **AWS EC2 Ubuntu** (20.04/22.04) with optional GPU instance (e.g., `g4dn.xlarge`).
- **Docker Engine** and **Docker Compose v2** installed.
- **AWS CLI** configured with appropriate IAM permissions for S3 and EC2 metadata.
- **Git** to clone and update submodules (if used).

---

## Directory Layout

```
stylegan2-docker/
├── datasets/                  # Put your cifar10.zip here
├── outputs/                   # Populated by GAN workloads
│   ├── train-auto/
│   ├── train-cifar/
│   ├── train-mixed/
│   ├── infer-standard/
│   └── infer-vector/
├── monitoring/                # Prometheus/Grafana/etc.
│   ├── dashboards/            # (optional) Grafana JSON files
│   ├── docker-compose.yml     # Monitoring services
│   └── prometheus.yml         # Scrape config
├── stylegan2-ada-pytorch/     # GAN code + Dockerfile + workloads compose
│   ├── Dockerfile
│   ├── docker-compose.yml     # Five StyleGAN2‑ADA services
│   ├── train.py
│   ├── generate.py
│   ├── dataset_tool.py
│   ├── ...                    # Other GAN scripts
│   └── upload_all_to_s3.sh    # Uploader script
└── README.md                  # This file
```

---

## Setup on EC2

1. **Clone the repo**
   ```bash
   git clone https://github.com/cgordon-dev/stylegan2-docker.git
   cd stylegan2-docker
   ```

2. **Install Docker & Compose**
   ```bash
   sudo apt update
   sudo apt install -y docker.io curl
   # Install Compose v2 plugin
   mkdir -p ~/.docker/cli-plugins
   curl -SL https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-$(uname -s)-$(uname -m) \
       -o ~/.docker/cli-plugins/docker-compose
   chmod +x ~/.docker/cli-plugins/docker-compose
   sudo systemctl enable --now docker
   docker --version && docker compose version
   ```

---

## Workload Execution

### Prepare Dataset

If you have `cifar10.zip`, place it under `datasets/`. Otherwise, generate it using the provided `prepare_cifar10.py` script:

```bash
# from stylegan2-ada-pytorch/
conda activate stylegan2-ada    # or your env
pip install torchvision pillow tqdm
chmod +x prepare_cifar10.py
./prepare_cifar10.py          # dumps cifar10_images/
python dataset_tool.py --source cifar10_images --dest ../datasets/cifar10.zip
```

### Start GAN Workloads

```bash
cd stylegan2-ada-pytorch
docker compose up --build -d
```

- **train-auto**: Standard training (`cfg=auto`).
- **train-cifar**: Fine‑tune on CIFAR‑10 (`cfg=cifar`).
- **train-mixed**: Mixed‑precision (`--fp32`).
- **infer-standard**: Image generation via `generate.py`.
- **infer-vector**: Vectorized inference via `generate_vectorized.py`.

Outputs are written to `../outputs/<workload>/`.

### Upload to S3

```bash
# from stylegan2-ada-pytorch/
chmod +x upload_all_to_s3.sh
./upload_all_to_s3.sh [optional_run_id]
```
This will sync `logs/`, `models/`, and `metrics/` to `s3://aws-gpu-monitoring-logs/...` with date, EC2 instance ID, and IP.

---

## Monitoring Stack

```bash
cd ../monitoring
docker compose up -d
```
- **Prometheus**: http://<EC2_IP>:9090
- **Grafana**: http://<EC2_IP>:3000 (default admin/admin)

Prometheus scrapes:
- Node Exporter (9100)
- DCGM Exporter (9400)
- cAdvisor (8080)
- GAN workloads (ports 8000 & 8001)

Import dashboards from `dashboards/` or Grafana community IDs:
- Node Exporter Full (ID 1860)
- Docker/cAdvisor (ID 893)
- NVIDIA DCGM Exporter

---

## Development & Contributing

- **Submodule vs. Vendoring**: If you prefer to track upstream changes of `stylegan2-ada-pytorch`, add it as a Git submodule. Otherwise, the code is vendored directly.
- **Pull Requests**: Feel free to open issues or PRs to improve workloads, monitoring, or uploader logic.

---

Enjoy training and monitoring StyleGAN2‑ADA end‑to‑end in containers! 🚀

