# 改某类功能时该去哪些文件

## 改 DSL 语法

例如新增关键字、语句、表达式：

- `lib/Frontend/Lexer.cpp`
- `lib/Frontend/Parser.cpp`
- `include/tiny-gpu-compiler/Frontend/AST.h`

如果语法影响 lowering，还要改：

- `lib/Frontend/MLIRGen.cpp`

测试优先补：

- `test/Frontend/*.tgc`

## 改 builtin 变量或 kernel 语义

例如新增 builtin，或调整 `threadIdx/blockIdx/blockDim` 行为：

- `include/.../Frontend/AST.h`
- `lib/Frontend/Parser.cpp`
- `lib/Frontend/MLIRGen.cpp`
- `lib/CodeGen/RegisterAllocator.cpp`
- 可能还要改 `include/.../Dialect/TinyGPU/TinyGPUOps.td`

## 改 IR / dialect

例如新增 `tinygpu.xxx` op：

- `include/tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.td`
- `lib/Dialect/TinyGPU/TinyGPUOps.cpp`
- `lib/Frontend/MLIRGen.cpp`
- `lib/CodeGen/TinyGPUEmitter.cpp`
- 可能还要改 `lib/CodeGen/RegisterAllocator.cpp`

## 改优化

- `include/tiny-gpu-compiler/Passes/Passes.h`
- `lib/Passes/Passes.cpp`
- `lib/Pipeline/Pipeline.cpp`（如果要调整接入顺序或开关）

测试优先补：

- `test/Frontend/*.tgc`
- 或新增更针对优化结果的 lit case

## 改寄存器分配

- `lib/CodeGen/RegisterAllocator.cpp`
- `lib/CodeGen/TinyGPUEmitter.cpp`

因为 emitter 假定 allocator 已经写好 `rd/rs/rt`。

## 改 ISA 编码 / 输出格式

- `lib/CodeGen/TinyGPUEmitter.cpp`
- `include/tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h`

如果是新增 op 对应编码，通常还要联动：

- `include/.../TinyGPUOps.td`
- `lib/Frontend/MLIRGen.cpp`
- `lib/CodeGen/RegisterAllocator.cpp`

## 改 web 可视化

UI 结构：

- `web/src/App.tsx`
- `web/src/components/*`

如果改 stages 展示：

- `web/src/components/PipelineView.tsx`

如果改 binary 展示：

- `web/src/components/BinaryView.tsx`

如果改分析面板：

- `web/src/components/AnalysisPanel.tsx`

## 改 web 编译行为

- `web/src/compiler/TGCCompiler.ts`
- `web/src/compiler/types.ts`

注意：这不会自动影响 native 编译器。

## 改 simulator 行为

- `web/src/simulator/TinyGPUSim.ts`

包括：

- barrier
- divergence
- memory
- thread stepping
- block dispatch

## 改测试/构建

native 测试：

- `test/lit.cfg.py`
- `test/CMakeLists.txt`

CI：

- `.github/workflows/ci.yml`

web 构建：

- `web/package.json`
- `.github/workflows/deploy-web.yml`
