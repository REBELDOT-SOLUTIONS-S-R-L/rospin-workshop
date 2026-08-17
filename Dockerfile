# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ARG TARGETARCH
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    ROSPIN_DATA_ROOT=/workspace/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglx-mesa0 \
        libglfw3 \
        libosmesa6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install CPU PyTorch first. Changing TORCH_INDEX_URL produces a CUDA training
# image without changing the application layer.
RUN python -m pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${TORCH_INDEX_URL}"

COPY pyproject.toml README.md ./
COPY src ./src
COPY tasks ./tasks
COPY trajectories ./trajectories
COPY assets /assets

RUN python -m pip install .

RUN echo "target architecture: ${TARGETARCH}" \
    && MUJOCO_GL=osmesa rospin-self-check

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=4 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=2)"

ENTRYPOINT ["rospin-serve"]
