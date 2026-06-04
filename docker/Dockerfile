# Modify these for other versions
ARG PYTORCH="2.12.0"
ARG CUDA="12.6"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends git ninja-build cmake build-essential \
     xterm xauth openssh-server tmux wget python3.12 python3.12-venv python3.12-dev\
    && rm -rf /var/lib/apt/lists/*

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive

# deps
RUN apt-get update

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

# PEP 668 workaround
RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED

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