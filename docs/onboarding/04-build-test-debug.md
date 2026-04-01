# 构建、测试与调试

## Native 构建

根入口：

- `CMakeLists.txt`

要求：

- CMake 3.20+
- C++17
- LLVM/MLIR 18
- Ninja 推荐

典型构建：

```bash
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm

cmake --build build
```

## CLI 用法

入口：

- `tools/tgc/tgc.cpp`

常见输出格式：

- `--emit mlir`
- `--emit asm`
- `--emit hex`
- `--emit bin`
- `--emit trace`

例子：

```bash
./build/bin/tgc --emit asm examples/vector_add.tgc
```

## 测试系统

测试配置：

- `test/CMakeLists.txt`
- `test/lit.cfg.py`

测试风格：

- LLVM lit
- `FileCheck`
- 测试文件本身是 `.tgc` 或 `.mlir`

示例：

- `test/Frontend/vector_add.tgc`
- `test/CodeGen/vector_add_asm.tgc`

你会看到典型模式：

```
// RUN: %tgc --emit mlir %s | FileCheck %s
// CHECK: ...
```

## CI

文件：

- `.github/workflows/ci.yml`

CI 做的事情：

1. checkout repo
2. 安装 CMake / Ninja / clang
3. 构建 LLVM/MLIR（缓存）
4. 构建 tiny-gpu-compiler
5. 跑 `check-tgc`
6. 编译 `examples/*.tgc`

这个 workflow 非常适合当“官方构建参考”。

## Web 构建

文件：

- `web/package.json`

命令：

```bash
cd web
npm install
npm run dev
npm run build
```

## Docker

文件：

- `Dockerfile`

适合你不想自己折腾 LLVM/MLIR 版本时使用。

## 调试建议

### 调 native compiler
优先看：

- `lib/Pipeline/Pipeline.cpp`
- `lib/Frontend/MLIRGen.cpp`
- `lib/CodeGen/RegisterAllocator.cpp`
- `lib/CodeGen/TinyGPUEmitter.cpp`

推荐方式：

1. 先跑一个最小 example
2. 分别看 `mlir` / `asm` / `trace`
3. 对照测试或 README 中的期望输出

### 调 web compiler
优先看：

- `web/src/compiler/TGCCompiler.ts`
- `web/src/simulator/TinyGPUSim.ts`
- `web/src/components/*`

推荐方式：

1. 固定一个 example kernel
2. 看 pipeline stages 是否和预期一致
3. 再看 binary 和 simulator state

## 一个值得留意的构建点

架构探索里发现：

- `lib/Passes/Passes.cpp` 存在并被 pipeline 使用
- 但 `lib/CMakeLists.txt` 的 wiring 值得新人第一时间验证

如果你遇到构建链接问题，优先检查 `Passes` 相关 target 是否正确接入。
