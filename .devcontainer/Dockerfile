# Stage 1: NVIDIA CUDA Image
ARG CUDA_VERSION=12.5.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS cuda-base

# Stage 2: Miniconda setup from configuration
FROM continuumio/miniconda3 AS miniconda-stage

# Stage 3: Final image combining CUDA and Miniconda
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Copy from CUDA base
COPY --from=cuda-base /usr/local/cuda /usr/local/cuda

# Copy Miniconda from the Miniconda stage
COPY --from=miniconda-stage /opt/conda /opt/conda

# Set environment variables for Miniconda
ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Python 3.10
ARG PYTHON_VERSION=3.12
RUN conda install python=${PYTHON_VERSION}

# Arguments for PyTorch and CUDA Toolkit versions
ARG PYTORCH_VERSION=2.5.0
ARG CUDATOOLKIT_VERSION=12.4

# Install PyTorch and other dependencies
RUN conda install pytorch=${PYTORCH_VERSION} pytorch-cuda=${CUDATOOLKIT_VERSION} -c pytorch -c nvidia

# Handle environment.yml if it exists
COPY environment.yml* noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
    /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; \
    fi \
    && rm -rf /tmp/conda-tmp

# Append Miniconda to PATH in .bashrc for interactive shells
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate base" >> /root/.bashrc

# Final CMD or ENTRYPOINT
CMD ["bash"]
