# StyleGAN2-ADA Docker Development Guide

This guide provides a comprehensive overview of the StyleGAN2-ADA GPU Docker project, helping new developers understand the project structure, setup procedures, and how to work with the codebase effectively.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup Requirements](#setup-requirements)
4. [Docker Configuration](#docker-configuration)
5. [Training Workflow](#training-workflow)
6. [Monitoring and Metrics](#monitoring-and-metrics)
7. [Development Workflow](#development-workflow)
8. [Troubleshooting](#troubleshooting)

## Project Overview

This project provides a Docker-based environment for training and using NVIDIA's StyleGAN2-ADA (Adaptive Discriminator Augmentation) model. StyleGAN2-ADA is a state-of-the-art generative adversarial network (GAN) that can produce high-quality images with limited training data.

Key features:
- Dockerized training environment for StyleGAN2-ADA
- GPU acceleration support
- Monitoring infrastructure with Prometheus and Grafana
- Automated training and evaluation pipeline
- Support for multi-stage training workflows

The implementation is based on NVIDIA's PyTorch port of StyleGAN2-ADA, providing a faithful reimplementation that focuses on correctness, performance, and compatibility.

## Repository Structure

The repository is organized into the following main components:

```
stylegan2ada-gpu-docker/
├── modified-stylegan2-ada-pytorch/ # Core StyleGAN2-ADA implementation
│   ├── training/                  # Training modules
│   ├── torch_utils/               # PyTorch utilities
│   ├── metrics/                   # Evaluation metrics
│   ├── dnnlib/                    # Deep learning utilities
│   ├── datasets/                  # Example datasets
│   ├── scripts/                   # Utility scripts
│   ├── Dockerfile                 # Main container definition
│   ├── docker-compose.yml         # Container orchestration
│   └── docker_run.sh              # Helper script for Docker execution
├── monitoring/                    # Monitoring infrastructure
│   ├── docker-compose.yml         # Monitoring stack configuration
│   └── prometheus.yml             # Prometheus configuration
└── README.md                      # Project documentation
```

## Setup Requirements

### Hardware Requirements

- NVIDIA GPU(s) with CUDA support
- Minimum 12GB of GPU memory (more recommended for higher resolutions)
- Sufficient disk space for datasets and generated outputs

### Software Requirements

- Linux or Windows (Linux recommended for performance)
- Docker and Docker Compose
- NVIDIA Container Toolkit (nvidia-docker2)
- NVIDIA GPU drivers (version r455.23 or later)

## Docker Configuration

The project uses Docker to provide an isolated and reproducible training environment. The main components are:

### Main Dockerfile

The Dockerfile in `modified-stylegan2-ada-pytorch/` defines the base container:
- Based on NVIDIA's PyTorch container (nvcr.io/nvidia/pytorch:20.12-py3)
- Installs required Python dependencies
- Configures the workspace for training

### Docker Compose Configuration

Two docker-compose files coordinate different aspects of the system:

1. **Training Compose (`modified-stylegan2-ada-pytorch/docker-compose.yml`)**:
   - Defines training services and their configurations
   - Manages dataset and output volumes
   - Provides execution parameters for different training scenarios

2. **Monitoring Compose (`monitoring/docker-compose.yml`)**:
   - Sets up Prometheus for metrics collection
   - Configures Grafana for visualization dashboards
   - Includes exporters for system, container, and GPU metrics

## Training Workflow

The training process in this project follows a multi-stage workflow:

1. **Dataset Preparation**:
   - Datasets are stored as uncompressed ZIP archives
   - Custom datasets can be created using `dataset_tool.py`

2. **Basic Training**:
   - Begins with the "auto" configuration for initial model training
   - Handles training parameter selection based on resolution and GPU count

3. **Fine-tuning**:
   - Resumes from previous snapshots for specialized training
   - Supports transfer learning from pre-trained models

4. **Mixed-precision Training**:
   - Uses mixed-precision for performance improvement
   - Adjusts batch sizes and other hyperparameters

5. **Image Generation**:
   - Creates sample images from trained models
   - Generates variation using different random seeds

6. **Model Evaluation**:
   - Computes quality metrics like FID, KID, and Precision/Recall
   - Evaluates model performance against benchmarks

7. **Results Handling**:
   - Uploads results to S3 (if configured)
   - Generates reports for easy model comparison

## Monitoring and Metrics

The project includes a comprehensive monitoring stack:

### Prometheus

- Collects metrics from training runs
- Monitors GPU utilization, memory usage, and training progress
- Provides time-series data for performance analysis

### Grafana

- Visualizes training metrics in real-time
- Customizable dashboards for different monitoring needs
- Accessible via port 3000 (default admin/admin credentials)

### Metrics Collectors

- **Node Exporter**: Host system metrics
- **DCGM Exporter**: NVIDIA GPU metrics
- **cAdvisor**: Container resource usage

## Development Workflow

When working on this project, follow these steps for an effective development workflow:

### Setting Up Your Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stylegan2ada-gpu-docker.git
   cd stylegan2ada-gpu-docker
   ```

2. Build the Docker images:
   ```bash
   cd modified-stylegan2-ada-pytorch
   docker build -t sg2ada:latest .
   cd ../monitoring
   docker-compose build
   ```

### Working with the Code

1. **Running Individual Commands**:
   Use the `docker_run.sh` script to execute commands in the container:
   ```bash
   ./docker_run.sh python generate.py --help
   ```

2. **Starting the Training Pipeline**:
   ```bash
   cd modified-stylegan2-ada-pytorch
   docker-compose up train-auto
   ```

3. **Starting the Monitoring Stack**:
   ```bash
   cd monitoring
   docker-compose up -d
   ```
   Then access Grafana at http://localhost:3000

### Using Pre-trained Models

To work with pre-trained models:

1. Download the desired model from NVIDIA's model collection:
   ```bash
   wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
   ```

2. Generate images:
   ```bash
   ./docker_run.sh python generate.py --outdir=out --trunc=0.7 --seeds=600-605 --network=ffhq.pkl
   ```

### Customizing Training

To customize training parameters:

1. Modify the parameters in the `docker-compose.yml` service definition
2. Use alternative configurations (stylegan2, paper256, etc.)
3. Adjust hyperparameters for specific datasets

## Troubleshooting

### Common Issues

1. **CUDA/GPU Problems**:
   - Ensure NVIDIA drivers are properly installed
   - Check that nvidia-docker2 is configured correctly
   - Verify GPU visibility with `nvidia-smi`

2. **Memory Errors**:
   - Reduce batch size for larger models
   - Use mixed precision training
   - Consider adjusting PYTORCH_CUDA_ALLOC_CONF

3. **Docker Issues**:
   - Ensure Docker daemon is running
   - Check network conflicts for exposed ports
   - Verify volume mounts are accessible

### Getting Help

- Consult the [StyleGAN2-ADA PyTorch documentation](https://github.com/NVlabs/stylegan2-ada-pytorch)
- Check container logs for specific error messages
- Use Grafana dashboards to identify performance bottlenecks

---

This guide should provide a comprehensive overview to get you started with the StyleGAN2-ADA GPU Docker project. For more detailed information on the StyleGAN2-ADA algorithm itself, refer to the [original research paper](https://arxiv.org/abs/2006.06676).