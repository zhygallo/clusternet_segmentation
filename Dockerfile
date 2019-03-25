FROM nvidia/cuda:9.1-devel-ubuntu16.04

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

#RUN /opt/conda/bin/conda install -c conda-forge tensorflow-gpu \
#                                                keras \
#                                                scikit-image \
#                                                click

RUN /opt/conda/bin/pip install sklearn \
                                scikit-image \
                                click \
                                pandas \
                                tensorboard==1.10.0 \
                                tensorflow-gpu==1.10.1 \
                                tensorlayer==1.10.1 \
                                Keras==2.2.2 \
                                Keras-Applications==1.0.6 \
                                Keras-Preprocessing==1.0.5

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH="/opt/conda/bin:${PATH}"
