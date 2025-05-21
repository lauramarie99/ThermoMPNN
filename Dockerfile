# Base image with CUDA 11.8 support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory inside the container
WORKDIR /workspaces

# Clone the ThermoMPNN repository
RUN git clone https://github.com/lauramarie99/ThermoMPNN.git
# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch + CUDA 11.8 wheels
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
RUN pip install \
    torchmetrics \
    pytorch-lightning \
    biopython \
    wandb \
    tqdm \
    pandas \
    numpy \
    omegaconf \
    joblib

# Install MMseqs2 GPU-optimized AVX2 build
RUN wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz && \
    tar xvfz mmseqs-linux-gpu.tar.gz && \
    mv mmseqs/bin/mmseqs /usr/local/bin/ && \
    rm -rf mmseqs mmseqs-linux-gpu.tar.gz

# Default entrypoint (can be overridden)
CMD ["python", "/workspaces/ThermoMPNN/analysis/SSM.py"]
