FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/hf-cache
ENV TRANSFORMERS_CACHE=/workspace/hf-cache
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    rsync \
    vim \
    nano \
    tmux \
    htop \
    openssh-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    transformers \
    accelerate \
    datasets \
    safetensors \
    sentencepiece \
    protobuf \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    jupyterlab \
    ipykernel \
    einops \
    tqdm \
    pyyaml \
    wandb \
    rich \
    ipywidgets \
    dtale

RUN python -m ipykernel install \
    --name persona_introspection \
    --display-name "persona_introspection"

WORKDIR /workspace/projects