# 建议阅读顺序

## 30 分钟速通版

如果你只想先抓住 repo 结构，按这个顺序读：

1. `README.md`
2. `lib/Pipeline/Pipeline.cpp`
3. `include/tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.td`
4. `lib/CodeGen/RegisterAllocator.cpp`
5. `lib/CodeGen/TinyGPUEmitter.cpp`
6. `include/tiny-gpu-compiler/Frontend/AST.h`
7. `lib/Frontend/Parser.cpp`
8. `lib/Frontend/MLIRGen.cpp`
9. `web/src/compiler/TGCCompiler.ts`
10. `web/src/simulator/TinyGPUSim.ts`
11. `test/lit.cfg.py`
12. `test/Frontend/vector_add.tgc`

## 为什么这样排

### 第一步：先看总编排
`lib/Pipeline/Pipeline.cpp`

这是全仓库最值得先读的文件，因为它定义了完整阶段边界：

- frontend
- MLIRGen
- optimization
- register allocation
- binary emission
- analysis

先看它，你会知道“每一层到底负责什么”。

### 第二步：看 IR 和硬件如何对齐
先看：

- `include/.../TinyGPUOps.td`
- `lib/CodeGen/RegisterAllocator.cpp`
- `lib/CodeGen/TinyGPUEmitter.cpp`

这三者定义了核心契约：

- dialect op 是什么
- op 怎么拿寄存器
- op 怎么变成 ISA

### 第三步：再回头看 frontend
看：

- `include/.../Frontend/AST.h`
- `lib/Frontend/Lexer.cpp`
- `lib/Frontend/Parser.cpp`
- `lib/Frontend/MLIRGen.cpp`

这样你不会陷在 parser 细节里，而是知道这些 frontend 工作最终要服务什么 IR/ISA。

### 第四步：把 web 当成独立实现看
看：

- `web/src/compiler/TGCCompiler.ts`
- `web/src/simulator/TinyGPUSim.ts`
- `web/src/examples/index.ts`

重点不是“它和 C++ 一模一样”，而是“它为了教学 UI 复刻了哪些概念”。

## 按角色阅读

### 如果你要改 DSL / 语法
先读：

- `include/tiny-gpu-compiler/Frontend/AST.h`
- `lib/Frontend/Parser.cpp`
- `lib/Frontend/MLIRGen.cpp`

### 如果你要改指令 / 后端
先读：

- `include/tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.td`
- `lib/CodeGen/RegisterAllocator.cpp`
- `lib/CodeGen/TinyGPUEmitter.cpp`

### 如果你要改优化
先读：

- `include/tiny-gpu-compiler/Passes/Passes.h`
- `lib/Passes/Passes.cpp`

### 如果你要改前端 UI / simulator
先读：

- `web/src/App.tsx`
- `web/src/components/*`
- `web/src/compiler/TGCCompiler.ts`
- `web/src/simulator/TinyGPUSim.ts`
