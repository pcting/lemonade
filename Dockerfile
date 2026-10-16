# ==============================================================
# # 1. Build stage — compile lemonade C++ binaries
# # ============================================================
FROM ubuntu:24.04 AS builder

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# npm must not prefer AAAA records: hosts behind IPv6-unrouted networks fail
# registry fetches with ENETUNREACH. A-record-first keeps working everywhere
# (IPv6 still used when it is actually reachable).
ENV NODE_OPTIONS=--dns-result-order=ipv4first

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libssl-dev \
    pkg-config \
    libdrm-dev \
    git \
    nodejs \
    npm \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app
WORKDIR /app

# Build the project
RUN rm -rf build && \
    cmake --preset default && \
    cmake --build --preset default web-app

# Debug: Check build outputs
RUN echo "=== Build directory contents ===" && \
    ls -la build/ && \
    echo "=== Checking for resources ===" && \
    find build/ -name "*.json" -o -name "resources" -type d

# ============================================================
# 1b. gufo engine builder — Strix Halo inference engine.
# Upstream publishes no binaries (the gufo backend is system-managed);
# the image bundles a pinned source build so PATH resolution finds it.
# ============================================================
FROM ubuntu:24.04 AS gufo-builder

# Pinned upstream release tag: the release the image bundles and the release the
# checkpoint compatibility manifest mirrors. The backend's minimum accepted
# version is a separate, lower constant (kMinGufoVersion), so bumping this does
# not raise the floor for engines already installed on a host. Override with a
# commit SHA for supply-chain-tight builds.
ARG GUFO_REPOSITORY=https://github.com/gufo-org/gufo.git
ARG GUFO_VERSION=v0.15.0

ENV DEBIAN_FRONTEND=noninteractive

# ROCm 7.2 apt repo (noble): HIP clang offloader + BLAS/rocWMMA dev headers.
# Ubuntu universe ships a stale Debian "ROCm" fork whose versions (5.7.1-3)
# outrank radeon's differently-versioned packages (hipcc is 1.1.1.70200!),
# dragging old /usr/include/hip headers into the build — prefer radeon and
# hard-block the fork's binary names.
RUN printf 'Package: *\nPin: origin "repo.radeon.com"\nPin-Priority: 900\n\nPackage: hipcc* libamdhip64* libamd-comgr* comgr* libhsa* rocminfo* rocm-device-libs*\nPin: origin "archive.ubuntu.com"\nPin-Priority: -1\n' \
        > /etc/apt/preferences.d/rocm-prefer-radeon \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 noble main" \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git pkg-config \
        libicu-dev libcurl4-openssl-dev libpng-dev libjpeg-turbo8-dev \
        libwebp-dev libssl-dev \
        rocm-core rocm-llvm rocm-llvm-dev rocm-device-libs hip-dev rocblas-dev hipblas-dev hipblaslt-dev \
        hipcub-dev rocprim-dev rocwmma-dev \
    && echo '/opt/rocm/lib' > /etc/ld.so.conf.d/rocm.conf && ldconfig \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${GUFO_VERSION}" "${GUFO_REPOSITORY}" /src/gufo && \
    git -C /src/gufo rev-parse HEAD

# Debian's ROCm deb layout keeps cmake's hip-config happy: hip-config.cmake
# resolves tools under ${_IMPORT_PREFIX}/bin, but the packages install
# hipcc/hipconfig to /usr/bin. hipcc is a wrapper that locates its perl
# sibling relative to $0, so a symlink breaks it — write exec-shims instead.
RUN for tool in hipcc hipconfig; do \
        if [ ! -e "/opt/rocm/bin/$tool" ] && command -v "$tool" >/dev/null; then \
            mkdir -p /opt/rocm/bin; \
            printf '#!/bin/sh\nexec "%s" "$@"\n' "$(command -v "$tool")" > "/opt/rocm/bin/$tool"; \
            chmod 755 "/opt/rocm/bin/$tool"; \
        fi; \
    done

WORKDIR /src/gufo
# The release preset turns HIP kernels on but not the offloader path; gfx1151
# code objects need ROCm's clang. The version/revision are release-pipeline
# cache vars (default "development"/"unknown") — inject them from the pin so
# `gufo --version` reports the pinned release. The --version call gates link +
# provenance before the runtime stage ever copies the artifact.
RUN cmake --preset release -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
        -DGUFO_RELEASE_VERSION="${GUFO_VERSION#v}" \
        -DGUFO_REVISION="$(git rev-parse --short HEAD)" && \
    cmake --build --preset release -j"$(nproc)" && \
    build/release/gufo --version

# # ============================================================
# # 2. Runtime stage — small, clean image
# # ============================================================
FROM ubuntu:24.04

# vLLM/Triton JIT-compiles native launcher modules at runtime.
# The gufo runtime package set is the engine's shared-library closure —
# validated by `gufo --version` inside the final image (2026-10-10 baseline
# against ROCm 7.2 on noble; adjust together with the builder's -dev set when
# bumping). ffmpeg/ffprobe are invoked by the engine itself.
# ffmpeg/ffprobe are invoked by the engine itself. The same radeon-preference
# pin guards the runtime set (hip-runtime-amd pulls rocminfo/hsa deps that the
# universe fork would otherwise win by version).
RUN printf 'Package: *\nPin: origin "repo.radeon.com"\nPin-Priority: 900\n\nPackage: hipcc* libamdhip64* libamd-comgr* comgr* libhsa* rocminfo* rocm-device-libs*\nPin: origin "archive.ubuntu.com"\nPin-Priority: -1\n' \
        > /etc/apt/preferences.d/rocm-prefer-radeon \
    && apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /usr/share/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 noble main" \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update && apt-get install -y \
    build-essential \
    libcurl4 \
    curl \
    libssl3 \
    zlib1g \
    libdrm2 \
    vulkan-tools \
    libvulkan1 \
    unzip \
    xz-utils \
    libgomp1 \
    libatomic1 \
    libreadline8 \
    rocm-core \
    hip-runtime-amd \
    rocblas \
    hipblas \
    hipblaslt \
    hsa-rocr \
    comgr \
    libicu74 \
    libwebp7 \
    libjpeg-turbo8 \
    libpng16-16t64 \
    ffmpeg \
    && echo '/opt/rocm/lib' > /etc/ld.so.conf.d/rocm.conf && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user; lemond never needs root at runtime.
RUN useradd -r -u 10001 -s /usr/sbin/nologin lemonade

# The application directory doubles as the user's HOME so Lemonade's cache and
# config directories (plus the HuggingFace cache) resolve to writable paths.
WORKDIR /opt/lemonade
ENV HOME=/opt/lemonade

# Provide a private runtime directory so lemond can use get_runtime_dir()
RUN mkdir -p /run/lemonade && chmod 700 /run/lemonade
ENV XDG_RUNTIME_DIR=/run/lemonade

# Copy built executables and resources from builder
COPY --from=builder /app/build/lemond ./lemond
COPY --from=builder /app/build/lemonade ./lemonade
COPY --from=builder /app/build/resources ./resources

# Bundled gufo engine: installed into the standard PATH so the system-managed
# resolver finds it with zero configuration. Operator overrides
# (gufo.server_bin / LEMONADE_GUFO_SERVER_BIN) still take precedence.
COPY --from=gufo-builder /src/gufo/build/release/gufo /usr/local/bin/gufo

# Seed the model registry so a freshly created config volume starts with the
# baked-in registrations (the gufo -Gufo rows among them). Docker only copies
# image content into a named volume on first mount, while it is empty; an
# existing config volume keeps its own user_models.json.
COPY --chown=lemonade:lemonade docker/user_models.json /opt/lemonade/.config/lemonade/user_models.json

# Make executables executable
RUN chmod +x ./lemond ./lemonade && chmod 755 /usr/local/bin/gufo && \
    /usr/local/bin/gufo --version

# Expose the lemond/lemonade binaries on PATH so `docker exec` users can run
# them (e.g. `lemonade list`, `lemonade pull`) without needing the full path.
ENV PATH="/opt/lemonade:${PATH}"

# Create cache directories and hand the whole tree to the unprivileged user.
RUN mkdir -p /opt/lemonade/llama/cpu \
    /opt/lemonade/llama/vulkan \
    /opt/lemonade/.config/lemonade \
    /opt/lemonade/.cache/huggingface \
    /opt/lemonade/.cache/lemonade && \
    chown -R lemonade:lemonade /opt/lemonade /run/lemonade

USER lemonade

# Expose default port
EXPOSE 13305

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:13305/live || exit 1

# Default command: start server in headless mode.
# Binds 0.0.0.0 because Docker port publishing (-p) reaches the container via
# its external interface, not loopback. Restrict exposure at run time by
# publishing to host loopback (-p 127.0.0.1:13305:13305) and/or setting
# LEMONADE_API_KEY. See docs/guide/install/docker.md.
CMD ["./lemond", "--host", "0.0.0.0"]
