# CPU training image for the ML-infra platform.
# Build:  docker build -f deploy/Dockerfile -t discovery-agents-ml .
# Run:    docker run --rm discovery-agents-ml train --smoke
#
# For GPU, switch the base image to an NVIDIA CUDA + PyTorch image and install the
# CUDA torch wheel; the training code selects nccl/cuda automatically.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISCOVERY_PROVIDER=mock

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package with the ML + Dask extras (CPU torch wheel).
RUN python -m pip install --upgrade pip && pip install ".[ml,dask]"

# Default to the platform CLI; override the subcommand at `docker run`.
ENTRYPOINT ["discovery-agents-ml"]
CMD ["train", "--smoke"]
