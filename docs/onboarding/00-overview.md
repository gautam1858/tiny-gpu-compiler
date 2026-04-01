# tiny-gpu-compiler 总览

## 它是什么

`tiny-gpu-compiler` 是一个教学型 GPU 编译器项目：把 `.tgc` 这种 C-like kernel DSL，编译成 `tiny-gpu` 硬件可执行的 16-bit 指令，并提供一个浏览器中的可视化编译/执行界面。

- GitHub: `gautam1858/tiny-gpu-compiler`
- 目标硬件: `adam-maj/tiny-gpu`
- 编译基础设施: MLIR / LLVM

## 这个 repo 最重要的理解方式

不要把它只看成“一个 MLIR demo”。

更准确的理解是：

1. **一个真实的 C++/MLIR 编译器**
2. **一个用于教学和可视化的 TypeScript 镜像实现**

这两套东西长得像，但**不是同一个实现**，也**不完全等价**。

## 两条主线

### 1. Native 编译器主线
目录：

- `include/tiny-gpu-compiler/`
- `lib/`
- `tools/tgc/`
- `test/`

职责：

- 词法/语法分析
- AST 构建
- 降低到 TinyGPU MLIR dialect
- 优化
- 寄存器分配
- 指令编码
- 输出 asm / hex / bin / trace

### 2. Web 可视化主线
目录：

- `web/src/compiler/`
- `web/src/simulator/`
- `web/src/components/`

职责：

- 在浏览器里重新实现一套编译流程
- 在前端做 GPU 执行模拟
- 展示 IR、binary、分析面板、线程状态

## 代码重心在哪里

这个 repo 的“脊柱”不是泛泛的 MLIR，而是：

**TinyGPU dialect + `rd/rs/rt` 属性契约 + 1:1 ISA emission**

也就是：

- dialect 定义“有哪些 IR op”
- register allocator 给 op 打上 `rd/rs/rt`
- emitter 读取这些属性并编码成 16-bit 指令

关键文件：

- `include/tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.td`
- `lib/CodeGen/RegisterAllocator.cpp`
- `lib/CodeGen/TinyGPUEmitter.cpp`

## 顶层目录地图

- `include/`：公共头文件与 dialect 声明
- `lib/`：核心实现
- `tools/tgc/`：CLI 入口
- `test/`：LLVM lit + FileCheck 测试
- `examples/`：示例 `.tgc`
- `web/`：React + TypeScript 可视化前端
- `.github/workflows/`：CI 与 web deploy
- `Dockerfile`：可复现构建环境

## 新人最先知道的事实

- 所有值基本围绕 **8-bit (`i8`)** 建模
- `tinygpu.func` **没有函数参数**
- kernel 参数通过**固定内存地址映射**传递
- special registers:
  - `R13 = blockIdx`
  - `R14 = blockDim`
  - `R15 = threadIdx`
- web 编译器不是调用 C++ 编译器，而是自己重新实现了一套
