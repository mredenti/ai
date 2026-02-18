# --- Stage 1: Build Environment ---
FROM python:3.11-slim AS builder

# Pinned Versions
ARG PYTORCH_VERSION=2.10.0
ARG MONARCH_VERSION=0.3.0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git cmake clang libunwind-dev \
    && rm -rf /var/lib/apt/lists/*

# Monarch 0.3.0 requires the Rust nightly toolchain for its messaging backend
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup toolchain install nightly && rustup default nightly

WORKDIR /install

# 1. Install Pinned PyTorch CPU
# The --index-url ensures we don't accidentally pull the 5GB+ CUDA binaries
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install Pinned Monarch (CPU-Only Mode)
# Setting USE_TENSOR_ENGINE=0 disables the RDMA/GPU-heavy components
ENV USE_TENSOR_ENGINE=0 uv sync
RUN pip install --no-cache-dir torchmonarch==${MONARCH_VERSION}

# --- Stage 2: Runtime Environment ---
FROM python:3.11-slim AS runtime

# Copy the installed libraries from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Runtime performance tuning for CPU
# Prevents "thread thundering" where PyTorch over-allocates CPU cores
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

WORKDIR /app

# Verification Script
RUN python3 -c "import torch; import monarch; \
    from importlib.metadata import version; \
    print(f'PyTorch: {torch.__version__}'); \
    print(f'Monarch: {version(\"torchmonarch\")}'); \
    print('Status: CPU Environment Ready')"

CMD ["python3"]
