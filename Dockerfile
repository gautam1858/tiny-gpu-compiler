FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake ninja-build clang lld \
    python3 python3-pip \
    git wget \
    && rm -rf /var/lib/apt/lists/*

# Build LLVM/MLIR from source (this takes ~30 minutes first time)
ARG LLVM_VERSION=18.1.8
WORKDIR /opt

RUN git clone --depth 1 --branch llvmorg-${LLVM_VERSION} \
    https://github.com/llvm/llvm-project.git

RUN cmake -G Ninja -S llvm-project/llvm -B llvm-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm-install \
    && cmake --build llvm-build --target install -j$(nproc)

# Build tiny-gpu-compiler
WORKDIR /opt/tiny-gpu-compiler
COPY . .

RUN git submodule update --init --recursive 2>/dev/null || true

RUN cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_DIR=/opt/llvm-install/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm-install/lib/cmake/llvm \
    && cmake --build build -j$(nproc)

# Runtime image (much smaller)
FROM ubuntu:22.04

COPY --from=builder /opt/tiny-gpu-compiler/build/bin/tgc /usr/local/bin/tgc
COPY --from=builder /opt/llvm-install/lib/lib*.so* /usr/local/lib/
COPY examples/ /opt/examples/

RUN ldconfig

WORKDIR /workspace
ENTRYPOINT ["tgc"]
