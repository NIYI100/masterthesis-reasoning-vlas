FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
ARG PYTHON_VERSION=3.10
ENV TORCH_CUDA_ARCH_LIST="Turing Ampere Ada Hopper"

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libegl1-mesa \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY usr_share_nvidia/ /usr/share/nvidia/

# Install Miniforge (conda with conda-forge by default)
RUN curl -L -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    chmod +x ~/miniforge.sh && \
    ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# Install ManiSkill2
RUN pip install mani-skill2==0.5.0 && \
    pip cache purge

# Install SimplerEnv
RUN cd ~ && git clone https://github.com/allenai/SimplerEnv --recurse-submodules && \
    pip install numpy==1.24.4 && \
    cd SimplerEnv/ManiSkill2_real2sim && \
    pip install -e . && \
    cd .. && \
    pip install -e .

RUN cd ~/SimplerEnv && pip install tensorflow==2.15.0 && \
    pip install -r requirements_full_install.txt && \
    pip install tensorflow[and-cuda]==2.15.1

RUN pip install git+https://github.com/nathanrooy/simulated-annealing && cd ~

ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

WORKDIR /root/SimplerEnv