FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Alternatively download nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04 manually with docker pull
# proxy needs to be provided from the outside

# need this for apt to not get stuck prompting the user for locale info
ARG DEBIAN_FRONTEND=noninteractive

ARG HOST_UNAME
ARG HOST_UID
ARG HOST_GID

# Install some basic utilities
RUN rm -f /etc/apt/sources.list.d/*.list \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone\
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    wget \
    sudo \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /deps

COPY docker/inputrc /root/.inputrc

SHELL ["/bin/bash", "-c"]

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
    
ENV PATH=/opt/conda/bin:$PATH


# bootstrap conda
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate base \
    && conda update -n base conda -y \
    && conda install python=3.11.9 \
    && python --version \
    && source /opt/conda/etc/profile.d/conda.sh \
    && conda init bash \
    && conda config --set auto_activate_base true

ENV FORCE_CUDA="1"


# We use cuda-12.4 by default
# RUN conda install cudatoolkit=12.4 -c conda-forge --yes \
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# add user of current host system
RUN groupadd -g ${HOST_GID} -o ${HOST_UNAME}
RUN useradd -l -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_UNAME}

# make sure conda env is always active on startup there as well
USER ${HOST_UNAME}


RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda init bash \
    && conda config --set auto_activate_base true

# switch back to root as default user
USER root


COPY requirements.txt .


RUN pip install -r requirements.txt


RUN curl -fsSL https://ollama.com/install.sh | sh


# copy inputrc to user for a nicer bash experience
RUN cp /root/.inputrc /home/${HOST_UNAME}/.inputrc \
    && chown ${HOST_UNAME}:${HOST_GID} /home/${HOST_UNAME}/.inputrc

WORKDIR /workspace