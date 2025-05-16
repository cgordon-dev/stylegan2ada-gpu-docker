# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# original ------------------------
# FROM nvcr.io/nvidia/pytorch:20.12-py3

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# RUN pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0

# WORKDIR /workspace

# # Copy all local files into the container
# COPY . /workspace

# # Unset TORCH_CUDA_ARCH_LIST and exec.  This makes pytorch run-time
# # extension builds significantly faster as we only compile for the
# # currently active GPU configuration.
# RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
# ENTRYPOINT ["/entry.sh"]
# original ------------------------

# Dockerfile (adjusted)

FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install \
      imageio-ffmpeg==0.4.3 \
      pyspng==0.1.0 \
      prometheus-client

WORKDIR /workspace

# Copy everything (including scripts/ and metrics/)
COPY . /workspace

# bring in the scripts & watcher
COPY scripts/          /workspace/scripts/
COPY metrics/          /workspace/metrics/


# Make the wrappers & scripts executable in their new locations
RUN chmod +x \
      /workspace/train_with_prometheus.py \
      /workspace/metrics/monitor.sh \
      /workspace/scripts/wait_for_snapshot.sh \
      /workspace/scripts/wait_for_snapshot_vector.sh

# NVIDIA entrypoint shim
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) \
 && chmod a+x /entry.sh

ENTRYPOINT ["/entry.sh"]