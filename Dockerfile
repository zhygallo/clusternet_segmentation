# NOTE: May need adjustment for your specific GPU/CUDA setup
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG FAISS_CPU_OR_GPU=cpu
ARG PYTHON_VERSION=3.10

ENV HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl software-properties-common bzip2 \
        python3 python3-pip gdal-bin libgdal-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda for FAISS
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda update -n base conda && \
    /opt/conda/bin/conda install -y -c pytorch faiss-${FAISS_CPU_OR_GPU} && \
    rm /tmp/conda.sh

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /workspace
COPY . /workspace
