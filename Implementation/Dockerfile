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
# Determine installed torch version and lock it to prevent upgrade
RUN python -c "import torch; print(f'torch=={torch.__version__}')" > constraints.txt && \
    cat constraints.txt && \
    grep -v "torch" requirements.txt > requirements.tmp && \
    pip install --no-cache-dir -r requirements.tmp -c constraints.txt && \
    rm requirements.tmp constraints.txt

# Reset frontend
ENV DEBIAN_FRONTEND=dialog

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command (can be overridden)
CMD ["bash"]
