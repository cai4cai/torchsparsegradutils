# ----- Base image: Ubuntu 22.04 + CUDA 12.4 -----
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ----- System setup & Python 3.12 -----
RUN apt-get update && \
    # 1 – core tools and ability to add PPAs
    apt-get install -y --no-install-recommends \
        software-properties-common curl gnupg2 sudo \
        git openssh-client && \
    # 2 – add the deadsnakes repo *before* we ask for python3.12*
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    # 3 – now 3.12 packages are visible
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    # 4 – bootstrap pip for 3.12
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 - && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ----- CUDA env -----
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# ----- Python deps -----
COPY environment.yml /tmp/req.txt
RUN python -m pip install --no-cache-dir -r /tmp/req.txt && \
    rm /tmp/req.txt

# ---------- Non‑root dev‑container user ----------
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd  --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME
    
CMD ["bash"]
    