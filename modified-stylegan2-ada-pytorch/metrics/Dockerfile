# metrics/Dockerfile

# 1) Base off your existing StyleGAN2‑ADA training image
FROM stylegan2-ada:train-auto

# 2) Set working directory
WORKDIR /workspace

# 3) Copy only the metrics scripts into the image
COPY metrics/monitor.sh   /metrics/monitor.sh
COPY metrics/calc_metrics.py  /metrics/calc_metrics.py

# 4) Install metric‑calculation dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    Pillow \
    pandas \
    tqdm \
    absl-py \
    tensorflow \
    imageio \
    opencv-python \
    requests

# 5) Make sure your monitoring script is executable
RUN chmod +x /metrics/monitor.sh

# 6) Default entrypoint to kick off your watcher
ENTRYPOINT ["bash","-lc","/metrics/monitor.sh"]