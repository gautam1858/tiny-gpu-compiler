# Native Build / Test / Run Guide

This document records the exact native build-and-run flow that worked in this environment for `tiny-gpu-compiler`.

It is written so you can:

1. start from a repo with no `build/` directory,
2. reuse the already-installed local LLVM/MLIR 18.1.8 toolchain,
3. rebuild the compiler yourself,
4. run tests yourself,
5. compile the example kernels yourself.

---

## 1. What this guide assumes

This guide assumes the following are already present on this machine:

- repo path: `/home/xuefeiz2/3rd/tiny-gpu-compiler`
- local LLVM source: `$HOME/llvm-project-18`
- local LLVM build dir: `$HOME/llvm-build-18`
- local LLVM install dir: `$HOME/llvm-install`

This guide also assumes the repo source already contains the MLIR 18.1.8 compatibility fixes that were made during debugging.

It does **not** assume Docker works. In this environment, Docker daemon access was blocked.

---

## 2. Why native build was used

The repo README recommends Docker first, but Docker was not usable in this environment because the current user could not access `/var/run/docker.sock`.

The native fallback matched the repo CI closely:

- LLVM/MLIR version: **18.1.8**
- generator: **Ninja**
- build type: **Release**
- explicit `MLIR_DIR` / `LLVM_DIR`

Reference points:

- `README.md`
- `.github/workflows/ci.yml`

---

## 3. Environment facts discovered during the successful run

### 3.1 Compiler paths

On this machine:

- `clang` exists
- `clang-20` exists
- `clang++-20` exists
- plain `clang++` was **not** available on `PATH`

So the configure command had to use explicit compiler paths:

```bash
-DCMAKE_C_COMPILER=/usr/lib/ccache/clang-20
-DCMAKE_CXX_COMPILER=/usr/lib/ccache/clang++-20
```

### 3.2 LLVM / MLIR install

The working MLIR install was:

```bash
$HOME/llvm-install/lib/cmake/mlir
$HOME/llvm-install/lib/cmake/llvm
```

Verification command:

```bash
"$HOME/llvm-install/bin/mlir-opt" --version
```

Expected result:

- reports `LLVM version 18.1.8`

### 3.3 FileCheck caveat

`check-tgc` requires `FileCheck`.

In this environment, `FileCheck` was available from the LLVM **build tree**:

```bash
$HOME/llvm-build-18/bin/FileCheck
```

So tests were run with that on `PATH`.

---

## 4. Optional: rebuild LLVM / MLIR 18.1.8 from scratch

You do **not** need to do this if `$HOME/llvm-install` already exists and is valid.

Only use this section if you want to reproduce the local LLVM/MLIR toolchain too.

### 4.1 Clone LLVM

```bash
git clone --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git "$HOME/llvm-project-18"
```

### 4.2 Configure LLVM / MLIR

```bash
cmake -G Ninja \
  -S "$HOME/llvm-project-18/llvm" \
  -B "$HOME/llvm-build-18" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-20 \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX="$HOME/llvm-install"
```

### 4.3 Build and install LLVM / MLIR

```bash
cmake --build "$HOME/llvm-build-18" --target install -j"$(nproc)"
```

### 4.4 Verify LLVM / MLIR install

```bash
test -f "$HOME/llvm-install/lib/cmake/mlir/MLIRConfig.cmake"
"$HOME/llvm-install/bin/mlir-opt" --version
```

---

## 5. Clean starting point for replaying the repo build

This is the state I am leaving you in after cleanup:

- repo source changes are kept
- local LLVM/MLIR install is kept
- repo-local `build/` directory is removed

That means you can replay the repo build from scratch without paying the LLVM rebuild cost.

---

## 6. Configure the repo natively

From the repo root:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/lib/ccache/clang-20 \
  -DCMAKE_CXX_COMPILER=/usr/lib/ccache/clang++-20 \
  -DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir" \
  -DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm"
```

Expected result:

- configure completes successfully
- build files are written under `build/`

---

## 7. Build the compiler

```bash
cmake --build build -j1
```

Why `-j1` here:

- it made debugging easier while iterating on MLIR compatibility issues
- after things are stable, you can switch to `-j$(nproc)` if you want

Expected result:

- `build/bin/tgc` is produced

Quick verification:

```bash
build/bin/tgc --help
```

---

## 8. Run native tests

Because `FileCheck` was not on default `PATH` in this environment, the successful test command was:

```bash
PATH="$HOME/llvm-build-18/bin:$PATH" cmake --build build --target check-tgc -j1
```

Expected result:

- `Passed: 2 (100.00%)`

If you run the test target **without** this PATH adjustment here, it may fail with:

```text
FileCheck: command not found
```

---

## 9. Smoke-compile all shipped examples

From repo root:

```bash
for f in examples/*.tgc; do
  echo "=== $f ==="
  build/bin/tgc --emit asm "$f"
done
```

The successful run compiled these 11 files:

- `examples/conv1d.tgc`
- `examples/dot_product.tgc`
- `examples/matrix_add.tgc`
- `examples/matrix_multiply.tgc`
- `examples/relu.tgc`
- `examples/saxpy.tgc`
- `examples/shared_reduce.tgc`
- `examples/shared_tile_add.tgc`
- `examples/vector_add.tgc`
- `examples/vector_max.tgc`
- `examples/vector_reduction.tgc`

Expected result:

- each file prints assembly successfully
- no `MLIRGen failed`
- no verifier errors

---

## 10. Minimal end-to-end replay commands

If you just want the shortest replay path from the cleaned repo state:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/lib/ccache/clang-20 \
  -DCMAKE_CXX_COMPILER=/usr/lib/ccache/clang++-20 \
  -DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir" \
  -DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm"

cmake --build build -j1

build/bin/tgc --help

PATH="$HOME/llvm-build-18/bin:$PATH" cmake --build build --target check-tgc -j1

for f in examples/*.tgc; do
  echo "=== $f ==="
  build/bin/tgc --emit asm "$f"
done
```

---

## 11. What was fixed in the repo during this debugging session

This section is here so you know what kinds of issues were solved before the native flow became stable.

### 11.1 MLIR 18 ODS / include drift

Fixed areas included:

- `FunctionInterfaces.td` include path moved from `mlir/IR/...` to `mlir/Interfaces/...`
- successor declarations for branch/jump updated to MLIR 18 ODS style
- `FunctionImplementation.h` include path moved to `mlir/Interfaces/...`

### 11.2 Builder API drift

Value-producing TinyGPU ops now needed explicit result types when created through `OpBuilder::create(...)`.

### 11.3 Passes and link integration

The optimization passes implementation existed but was not wired into the CMake build graph, so a dedicated `TGCPasses` library was added and linked into the pipeline.

### 11.4 TinyGPU dialect link fix

The generated dialect declared `parseType` / `printType` overrides, so stub implementations had to be provided because the dialect defines no custom types.

### 11.5 MLIRGen runtime / CFG fixes

Two runtime issues were fixed:

- avoid calling `addEntryBlock()` when the function already has one
- ensure `if` / `for` lowering produces valid control flow and final terminators

---

## 12. Troubleshooting notes

### Problem: `clang++` not found

Use the explicit compiler paths from this guide instead of plain `clang++`.

### Problem: `MLIRConfig.cmake` not found

Check:

```bash
ls "$HOME/llvm-install/lib/cmake/mlir"
ls "$HOME/llvm-install/lib/cmake/llvm"
```

### Problem: `FileCheck: command not found`

Use:

```bash
PATH="$HOME/llvm-build-18/bin:$PATH" cmake --build build --target check-tgc -j1
```

### Problem: Docker instructions from README do not work

That is expected in this environment unless the current user is granted Docker daemon access.

---

## 13. What I cleaned for your replay

Per your request, cleanup is limited to **repo-local build artifacts only**.

I am **not** removing:

- `$HOME/llvm-install`
- `$HOME/llvm-build-18`
- `$HOME/llvm-project-18`
- repo source changes
- `.sisyphus/` notes

I **am** removing:

- repo `build/` directory

That leaves you with a clean repo build state but preserves the expensive LLVM setup.
