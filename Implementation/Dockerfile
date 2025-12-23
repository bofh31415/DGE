# Base image with PyTorch 2.1 and CUDA 11.8
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    htop \
    vim \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/DGE

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
# IMPORTANT: Filter out torch to verify we use the base image's version (2.1.0)
# and avoid pulling in massive CUDA 12 wheels.
RUN grep -v "torch" requirements.txt > requirements.tmp && \
    pip install --no-cache-dir -r requirements.tmp && \
    rm requirements.tmp

# Reset frontend
ENV DEBIAN_FRONTEND=dialog

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command (can be overridden)
CMD ["bash"]
