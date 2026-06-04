# Modify these for other versions
ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/shwoo1219/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"
ENV DEBIAN_FRONTEND=noninteractive

# deps
RUN apt-get update
RUN apt-get install -y --no-install-recommends git  cmake build-essential \
    libopenblas-dev xterm xauth openssh-server tmux wget \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace
RUN rm -rf MinkowskiEngine
COPY requirements.txt /workspace/requirements.txt

# ---- Jupyter + CV / training tooling ---------------------------------------
RUN python -m pip install --no-cache-dir \
        -r /workspace/requirements.txt 

RUN rm -rf requirements.txt

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