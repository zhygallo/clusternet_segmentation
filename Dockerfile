FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

ARG FAISS_CPU_OR_GPU=cpu
ARG FAISS_VERSION=1.4.0

ENV HOME /root

RUN apt-get update && \
    apt-get install -y build-essential curl software-properties-common bzip2 python3-pip  && \
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda update -n base conda && \
    /opt/conda/bin/conda install -y -c pytorch faiss-${FAISS_CPU_OR_GPU}=${FAISS_VERSION} && \
    apt-get remove -y --auto-remove curl bzip2 && \
    apt-get clean && \
    rm -fr /tmp/conda.sh

RUN /opt/conda/bin/conda install -c conda-forge tensorflow-gpu \
                                                keras \
                                                scikit-image \
                                                click


ENV PATH="/opt/conda/bin:${PATH}"
