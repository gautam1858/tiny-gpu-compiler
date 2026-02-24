# tiny-gpu-compiler

**An MLIR-based compiler that takes GPU kernels and compiles them to real hardware instructions, step by step.**

Most engineers know GPUs run parallel code. Few understand exactly what happens between writing `c[i] = a[i] + b[i]` and the silicon executing it. This project makes that journey visible. Write a GPU kernel in a C-like language, watch it lower through intermediate representations, see 16-bit binary instructions get emitted, and then run those instructions on a cycle-accurate GPU simulator -- all in your browser.

Built on [MLIR](https://mlir.llvm.org/) (the compiler infrastructure behind TensorFlow, PyTorch, and CUDA), targeting [tiny-gpu](https://github.com/adam-maj/tiny-gpu)'s open-source Verilog hardware.

<p align="center">
  <img src="docs/images/demo.gif" alt="tiny-gpu-compiler: compile, visualize, and execute GPU kernels" width="100%">
</p>

<p align="center">
  <a href="https://gautam1858.github.io/tiny-gpu-compiler/">Live Demo</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#the-compilation-pipeline">How It Works</a> &middot;
  <a href="#the-dsl">Language Reference</a>
</p>

---

## Why This Exists

[tiny-gpu](https://github.com/adam-maj/tiny-gpu) teaches how GPU **hardware** works at the RTL level. [tinygrad](https://github.com/tinygrad/tinygrad) teaches how GPU **software frameworks** work. Neither covers the critical middle layer: **the compiler that bridges them**.

This is the missing piece. It shows the full compilation pipeline from high-level parallel code to machine instructions that execute on actual GPU hardware:

```
Source Code (.tgc)          MLIR IR (TinyGPU Dialect)           16-bit Binary

kernel vec_add(             tinygpu.func @vec_add() {           0: 0x50DE  MUL R0,
  global int* a,              %0 = tinygpu.block_id : i8                   %blockIdx,
  global int* b,              %1 = tinygpu.block_dim : i8                  %blockDim
  global int* c) {     -->    %2 = tinygpu.mul %0, %1 : i8 --> 1: 0x300F  ADD R0,
  int i = blockIdx *          %3 = tinygpu.thread_id : i8                  R0,
          blockDim            %4 = tinygpu.add %2, %3 : i8                 %threadIdx
          + threadIdx;        %5 = tinygpu.load %4 : i8        2: 0x7100  LDR R1, [R0]
  c[i] = a[i] + b[i];        ...                               ...
}                             tinygpu.ret                       7: 0xF000  RET
                            }
```

---

## Quick Start

### Interactive Visualizer (No Installation Required)

Open the **[live demo](https://gautam1858.github.io/tiny-gpu-compiler/)** in your browser. Select an example kernel, step through the compilation stages, and watch the GPU execute your code cycle by cycle.

### Command-Line Compiler

```bash
# Clone
git clone --recursive https://github.com/gautam1858/tiny-gpu-compiler
cd tiny-gpu-compiler

# Docker (recommended -- includes LLVM/MLIR pre-built)
docker build -t tgc .
docker run -v $(pwd)/examples:/workspace tgc --emit asm /workspace/vector_add.tgc

# Native build (requires LLVM/MLIR 18)
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm
cmake --build build
./build/bin/tgc --emit asm examples/vector_add.tgc
```

Output:

```asm
; vector_add -- 11 instructions, 3 registers used
0x50DE  MUL R0, %blockIdx, %blockDim   ; idx = blockIdx * blockDim
0x300F  ADD R0, R0, %threadIdx          ; idx += threadIdx
0x7100  LDR R1, [R0]                    ; R1 = a[idx]
0x9240  CONST R2, #64                   ; R2 = base address of b
0x3220  ADD R2, R2, R0                  ; R2 = &b[idx]
0x7220  LDR R2, [R2]                    ; R2 = b[idx]
0x3112  ADD R1, R1, R2                  ; R1 = a[idx] + b[idx]
0x9280  CONST R2, #128                  ; R2 = base address of c
0x3220  ADD R2, R2, R0                  ; R2 = &c[idx]
0x8021  STR [R2], R1                    ; c[idx] = R1
0xF000  RET                             ; done
```

---

## The Compilation Pipeline

The compiler performs four distinct transformations, each visible in the interactive visualizer:

```
     .tgc Source Code
            |
            v
   +------------------+
   |  Lexer + Parser  |     Tokenizes source, builds an AST via
   +---------+--------+     recursive descent with precedence climbing
             |
             v
   +------------------+
   |     MLIRGen      |     Walks the AST, emits TinyGPU dialect
   +---------+--------+     operations (all values are i8)
             |
             v
   +------------------+
   |  Register Alloc  |     Linear scan over 13 general-purpose
   +---------+--------+     registers (R0-R12). R13/R14/R15 are
             |               reserved for blockIdx/blockDim/threadIdx
             v
   +------------------+
   |  Binary Emitter  |     Encodes each MLIR op into a 16-bit
   +---------+--------+     instruction word matching tiny-gpu's ISA
             |
             v
       16-bit Binary
       (runs on tiny-gpu
        Verilog hardware)
```

### Stage 1: Source Code

A C-like DSL designed for readability. Built-in variables `threadIdx`, `blockIdx`, and `blockDim` map directly to hardware registers.

```c
kernel vector_add(global int* a, global int* b, global int* c) {
    int idx = blockIdx * blockDim + threadIdx;
    c[idx] = a[idx] + b[idx];
}
```

### Stage 2: MLIR Intermediate Representation

The frontend emits operations in the `tinygpu` dialect. Every value is `i8` (8-bit unsigned), matching the hardware's data path. Pointer parameters are mapped to fixed 64-byte regions in data memory (a: 0-63, b: 64-127, c: 128-191).

<p align="center">
  <img src="docs/images/mlir-view.png" alt="MLIR TinyGPU dialect intermediate representation" width="100%">
</p>

### Stage 3: Register Allocation

A linear scan allocator assigns physical registers. The IR is annotated with `{rd=N, rs=M, rt=K}` attributes showing which hardware registers each operation uses.

<p align="center">
  <img src="docs/images/register-allocation.png" alt="Register allocation annotations on MLIR IR" width="100%">
</p>

### Stage 4: Binary Emission + Execution

Each instruction is encoded into a 16-bit word. The binary view color-codes each field: opcode (red), destination register (blue), source registers (green, orange). The GPU simulator then executes these instructions across parallel threads.

<p align="center">
  <img src="docs/images/simulation-complete.png" alt="GPU simulation complete with correct memory results" width="100%">
</p>

---

## The DSL

A minimal language for expressing GPU kernels:

```c
kernel matrix_multiply(global int* A, global int* B, global int* C, int N) {
    int idx = blockIdx * blockDim + threadIdx;
    int row = idx / N;
    int col = idx - row * N;
    int sum = 0;
    for (int k = 0; k < N; k = k + 1) {
        int a_val = A[row * N + k];
        int b_val = B[k * N + col];
        sum = sum + a_val * b_val;
    }
    C[idx] = sum;
}
```

| Feature | Details |
|---------|---------|
| **Types** | `int` (8-bit unsigned, matching hardware), `global int*` (pointer to data memory) |
| **Built-ins** | `threadIdx`, `blockIdx`, `blockDim` -- map to hardware registers R15, R13, R14 |
| **Arithmetic** | `+`, `-`, `*`, `/` |
| **Comparisons** | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| **Control flow** | `for` loops, `if`/`else` |
| **Memory** | Array indexing with `[]` on pointer parameters |

### Memory Layout

Pointer parameters are mapped to contiguous 64-byte regions in the hardware's 256-byte data memory:

| Parameter | Memory Region |
|-----------|--------------|
| 1st `global int*` | Addresses 0-63 |
| 2nd `global int*` | Addresses 64-127 |
| 3rd `global int*` | Addresses 128-191 |
| Scalar `int` parameters | Addresses 192+ |

---

## Target Hardware

The compiler emits binary for [tiny-gpu](https://github.com/adam-maj/tiny-gpu)'s 16-bit instruction set architecture:

### ISA Encoding

Each instruction is 16 bits wide, divided into four 4-bit fields:

```
  15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0
 +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 |   opcode      |     rd/nzp    |      rs       |      rt       |
 +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
```

### Instruction Set

| Opcode | Mnemonic | Encoding | Operation |
|--------|----------|----------|-----------|
| `0000` | NOP | `0000 ---- ---- ----` | No operation |
| `0001` | BRnzp | `0001 nzp- target--` | Branch if (flags & nzp) != 0 |
| `0010` | CMP | `0010 ---- rs-- rt--` | Set NZP flags from rs - rt |
| `0011` | ADD | `0011 rd-- rs-- rt--` | rd = rs + rt |
| `0100` | SUB | `0100 rd-- rs-- rt--` | rd = rs - rt |
| `0101` | MUL | `0101 rd-- rs-- rt--` | rd = rs * rt |
| `0110` | DIV | `0110 rd-- rs-- rt--` | rd = rs / rt |
| `0111` | LDR | `0111 rd-- rs-- ----` | rd = mem[rs] |
| `1000` | STR | `1000 ---- rs-- rt--` | mem[rs] = rt |
| `1001` | CONST | `1001 rd-- imm-----` | rd = 8-bit immediate |
| `1111` | RET | `1111 ---- ---- ----` | Thread done |

### Register File

| Register | Purpose |
|----------|---------|
| R0-R12 | General purpose (13 registers, managed by linear scan allocator) |
| R13 | `%blockIdx` -- current block index (read-only) |
| R14 | `%blockDim` -- threads per block (read-only) |
| R15 | `%threadIdx` -- thread index within block (read-only) |

### Hardware Specifications

| Parameter | Value |
|-----------|-------|
| Data width | 8 bits |
| Instruction width | 16 bits |
| Data memory | 256 bytes |
| Program memory | 256 entries |
| Execution model | SIMD lockstep within blocks, sequential block dispatch |

---

## TinyGPU MLIR Dialect

The compiler defines a custom MLIR dialect with 15 operations, each mapping directly to hardware capabilities:

| Operation | Signature | Hardware Mapping |
|-----------|-----------|------------------|
| `tinygpu.func` | `@name() { ... }` | Kernel entry point |
| `tinygpu.thread_id` | `-> i8` | R15 read |
| `tinygpu.block_id` | `-> i8` | R13 read |
| `tinygpu.block_dim` | `-> i8` | R14 read |
| `tinygpu.add` | `i8, i8 -> i8` | ADD opcode |
| `tinygpu.sub` | `i8, i8 -> i8` | SUB opcode |
| `tinygpu.mul` | `i8, i8 -> i8` | MUL opcode |
| `tinygpu.div` | `i8, i8 -> i8` | DIV opcode |
| `tinygpu.load` | `i8 -> i8` | LDR opcode |
| `tinygpu.store` | `i8, i8 -> ()` | STR opcode |
| `tinygpu.const` | `attr -> i8` | CONST opcode |
| `tinygpu.cmp` | `i8, i8 -> i8` | CMP opcode (sets NZP flags) |
| `tinygpu.branch` | `i8, attr, successor` | BRnzp opcode |
| `tinygpu.jump` | `successor` | BRnzp with mask=0b111 |
| `tinygpu.ret` | `-> ()` | RET opcode |

The dialect is defined in [TableGen ODS](include/tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.td), the standard MLIR approach for declarative operation specification.

---

## Output Formats

```bash
# MLIR intermediate representation (TinyGPU dialect)
tgc --emit mlir examples/vector_add.tgc

# Annotated assembly with hex encoding
tgc --emit asm examples/vector_add.tgc

# Raw hex (loadable into tiny-gpu Verilog testbench)
tgc --emit hex examples/vector_add.tgc

# Raw binary
tgc --emit bin examples/vector_add.tgc

# JSON trace (consumed by the web visualizer)
tgc --emit trace examples/vector_add.tgc
```

---

## Examples

| Kernel | Description | Instructions | Key Concepts |
|--------|-------------|-------------|--------------|
| [`vector_add.tgc`](examples/vector_add.tgc) | `c[i] = a[i] + b[i]` | 11 | Basic parallel kernel, memory addressing |
| [`dot_product.tgc`](examples/dot_product.tgc) | `c[i] = a[i] * b[i]` | 11 | Per-element multiply for reduction |
| [`saxpy.tgc`](examples/saxpy.tgc) | `y[i] = a * x[i] + y[i]` | 14 | BLAS Level 1, scalar parameter loading |
| [`relu.tgc`](examples/relu.tgc) | `max(0, input[i])` | 15 | Conditional branching (if/else), neural network activation |
| [`vector_max.tgc`](examples/vector_max.tgc) | `max(a[i], b[i])` | 16 | Element-wise comparison with control flow |
| [`conv1d.tgc`](examples/conv1d.tgc) | 1D convolution with sliding kernel | 22 | For-loops, accumulation, register reuse |
| [`matrix_add.tgc`](examples/matrix_add.tgc) | Element-wise matrix addition | 11 | Same as vector_add, different interpretation |
| [`matrix_multiply.tgc`](examples/matrix_multiply.tgc) | Full matrix multiply with loop | 28 | For-loops, multi-register allocation |
| [`vector_reduction.tgc`](examples/vector_reduction.tgc) | Conditional filtering | ~15 | If/else control flow |

---

## Project Structure

```
tiny-gpu-compiler/
  include/tiny-gpu-compiler/
    Dialect/TinyGPU/
      TinyGPUDialect.td          # Dialect definition (TableGen)
      TinyGPUOps.td              # 15 operations (TableGen ODS)
      TinyGPUDialect.h           # Generated dialect interface
      TinyGPUOps.h               # Generated operation interfaces
    Frontend/
      Lexer.h                    # Token types and lexer interface
      Parser.h                   # Recursive descent parser
      AST.h                      # Abstract syntax tree nodes
      MLIRGen.h                  # AST to MLIR conversion
    CodeGen/
      RegisterAllocator.h        # Linear scan register allocator
      TinyGPUEmitter.h           # Binary emission and trace output
    Pipeline/
      Pipeline.h                 # End-to-end compilation orchestration
  lib/
    Dialect/TinyGPU/             # Dialect and operation implementations
    Frontend/                    # Lexer, parser, MLIR generation
    CodeGen/                     # Register allocation, binary emission
    Pipeline/                    # Pipeline driver
  tools/tgc/
    tgc.cpp                      # Command-line compiler driver
  web/
    src/
      compiler/TGCCompiler.ts    # In-browser compiler (TypeScript)
      simulator/TinyGPUSim.ts    # Cycle-accurate GPU simulator
      components/
        Editor.tsx               # Monaco editor with .tgc highlighting
        PipelineView.tsx         # Compilation stage viewer
        BinaryView.tsx           # Color-coded binary instruction view
        GPUSimulator.tsx         # Interactive GPU execution viewer
      examples/index.ts          # Pre-loaded example kernels
  examples/                      # .tgc source files
  test/                          # LLVM lit tests
  Dockerfile                     # Reproducible build with LLVM/MLIR
```

---

## Building from Source

### Prerequisites

- CMake 3.20+
- C++17 compiler (Clang or GCC)
- Ninja (recommended)
- LLVM/MLIR 18

### Option 1: Docker (Recommended)

The Dockerfile builds LLVM/MLIR from source and compiles the project in a single reproducible step:

```bash
docker build -t tgc .
docker run tgc --emit asm /workspace/vector_add.tgc
```

### Option 2: Build LLVM/MLIR Locally

```bash
git clone --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git
cmake -G Ninja -S llvm-project/llvm -B llvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install
cmake --build llvm-build --target install
```

Then build tiny-gpu-compiler:

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-install/lib/cmake/llvm
cmake --build build
```

### Running Tests

```bash
cmake --build build --target check-tgc
```

### Web Visualizer (Development)

```bash
cd web
npm install
npm run dev
```

---

## Roadmap

- [x] TinyGPU MLIR dialect (15 operations defined in TableGen)
- [x] Frontend compiler (lexer, parser, AST, MLIR generation)
- [x] Register allocator (linear scan, 13 GPRs)
- [x] Binary emitter (16-bit ISA encoding)
- [x] Interactive web visualizer with in-browser compiler and GPU simulator
- [ ] CIRCT backend (generate custom accelerator Verilog via `tinygpu` -> `hw` + `comb` + `seq` lowering)
- [ ] WASM compilation (run the C++ MLIR compiler entirely in the browser via Emscripten)
- [ ] Hardware architecture overlay (show datapath schematic alongside execution)

---

## How It Relates to Real GPU Compilers

This project implements a simplified version of what production GPU compilers do:

| Concept | tiny-gpu-compiler | Production (e.g., NVIDIA CUDA) |
|---------|-------------------|-------------------------------|
| **IR Framework** | MLIR (TinyGPU dialect) | LLVM IR / NVVM / PTX |
| **Register Allocation** | Linear scan, 13 registers | Graph coloring, thousands of registers |
| **Memory Model** | Flat 256-byte address space | Global/shared/local/constant memory hierarchy |
| **Instruction Width** | 16-bit fixed | 64-bit+ variable-length |
| **Thread Model** | Lockstep SIMD within blocks | Warps of 32 threads, independent scheduling |
| **Data Width** | 8-bit unsigned | 32/64-bit float and integer |

The fundamental compilation stages are the same. The simplifications make each stage understandable without losing the essential structure.

---

## Acknowledgments

- [tiny-gpu](https://github.com/adam-maj/tiny-gpu) by Adam Majmudar -- the Verilog hardware that this compiler targets
- [MLIR](https://mlir.llvm.org/) / [LLVM Project](https://llvm.org/) -- the compiler infrastructure foundation
- [CIRCT](https://github.com/llvm/circt) -- hardware compiler framework (planned for Phase 3)

## License

Apache 2.0 with LLVM Exceptions. See [LICENSE](LICENSE).
