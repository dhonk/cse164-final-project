###############################################################################
# CUDA 12.6 + PyTorch 2.12 + torchvision + MinkowskiEngine
#
# Based on the CiSong10 cuda12-installation reference Dockerfile, retargeted
# from PyTorch 2.7.0 to 2.12.0 and extended with Jupyter + train/infer tooling.
###############################################################################

# Modify these for other versions
ARG PYTORCH="2.12.0"
ARG CUDA="12.6"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# ARG CUDA_SM_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9 9.0 12.0"
# I have a 3090, change depending on what you need
ENV TORCH_CUDA_ARCH_LIST="8.6"

ENV DEBIAN_FRONTEND=noninteractive

# deps
RUN apt-get update
RUN apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends git ninja-build cmake build-essential \
    libopenblas-dev xterm xauth openssh-server tmux wget gcc-11 g++-11 \
    python3.12 python3.12-venv python3.12-dev\
    && rm -rf /var/lib/apt/lists/*

# set gcc 11 to default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11

# set python3.12 to default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 100

# PEP 668 workaround
RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED

# build
WORKDIR /workspace
RUN git clone https://github.com/CiSong10/MinkowskiEngine.git \
    && cd MinkowskiEngine \
    && git checkout cuda12-installation

WORKDIR /workspace/MinkowskiEngine

# Fix for NVIDIA/MinkowskiEngine#596: the compiler finds both std::__to_address
# and cuda::std::__to_address. Patch libstdc++'s header to disambiguate.
# Path targets gcc-11 headers, which is what the Ubuntu 22.04 base ships.
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' \
        /usr/include/c++/11/bits/shared_ptr_base.h

# OOM issues
# ENV MAX_JOBS=2
# ENV MAKEFLAGS="-j2"

# --force_cuda is essential: there is no GPU visible during `docker build`,
# so without it setup.py would build the CPU-only variant.
RUN python setup.py install --force_cuda --blas=openblas

# Sanity-check the import at build time (CUDA runtime is only testable at
# `docker run`, but a failed import surfaces a broken build immediately).
RUN python -c "import MinkowskiEngine as ME; print('MinkowskiEngine', ME.__version__)"

WORKDIR /workspace
RUN rm -rf MinkowskiEngine
COPY requirements.txt /workspace/requirements.txt

# ---- Jupyter + CV / training tooling ---------------------------------------
RUN python -m pip install --no-cache-dir \
        -r /workspace/requirements.txt 

# ---- Runtime ---------------------------------------------------------------
WORKDIR /workspace
EXPOSE 8888

# Dispatch tasks via docker/entrypoint.sh: `docker compose run cse164 <task>`.
# Default task is `notebook` (JupyterLab, auth disabled for LOCAL convenience;
# re-enable a token if you expose 8888 beyond localhost — see README).
# The `sed` strips CRLF so the script runs even if checked out on Windows.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["help"]