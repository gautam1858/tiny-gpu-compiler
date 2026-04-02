# Native 编译器架构

## 入口

CLI 入口：

- `tools/tgc/tgc.cpp`

核心调用：

- 解析命令行
- 读取 `.tgc`
- 构造 `CompilerOptions`
- 调用 `tgc::compile(...)`

公共 API：

- `include/tiny-gpu-compiler/Pipeline/Pipeline.h`

## 编译阶段

在 `lib/Pipeline/Pipeline.cpp` 中，主流程大致是：

1. `Lexer`
2. `Parser`
3. `mlirGen`
4. `runAllOptimizations`
5. `allocateRegisters`
6. `emitBinary`
7. 输出 assembly / hex / bin / json trace

## Frontend

### AST
文件：

- `include/tiny-gpu-compiler/Frontend/AST.h`

这里定义：

- `ExprKind`
- `StmtKind`
- `KernelDef`
- `Program`

重点支持的概念：

- builtin vars: `threadIdx`, `blockIdx`, `blockDim`
- array index / shared array index
- `for`
- `if/else`
- shared memory declaration
- `__syncthreads()`

### Lexer / Parser
文件：

- `lib/Frontend/Lexer.cpp`
- `lib/Frontend/Parser.cpp`

风格：

- 手写 lexer
- recursive descent parser
- precedence climbing 处理表达式优先级

### MLIRGen
文件：

- `lib/Frontend/MLIRGen.cpp`

这是 frontend 最关键的实现文件。

它负责把 AST 直接降低到 `tinygpu` dialect，而不是经过更通用的 GPU dialect。

## 最重要的 lowering 约定

### 1. 参数不是函数参数，而是 memory-mapped
`tinygpu.func` 创建时没有参数。

参数映射规则：

- 第 1 个 `global int*` → base `0`
- 第 2 个 `global int*` → base `64`
- 第 3 个 `global int*` → base `128`
- scalar `int` 参数 → 从地址 `192` 往后放

这意味着 kernel 的“参数传递”本质上是固定地址协议。

### 2. shared memory 是顺序打包的
shared arrays 在 `MLIRGen.cpp` 里先做一遍收集，然后按声明顺序分配 offset。

这很重要，因为 shared memory 布局不是动态的，也没有复杂 allocator。

### 3. control flow 直接映射成 TinyGPU blocks/branches
`for` / `if` 会被降低成 block + branch/jump 形式，后续 emitter 再把 block 地址编码成绝对指令地址。

## Dialect

文件：

- `include/tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.td`
- `include/tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.td`
- `lib/Dialect/TinyGPU/TinyGPUDialect.cpp`
- `lib/Dialect/TinyGPU/TinyGPUOps.cpp`

这是一个自定义 MLIR dialect，核心思想不是“尽量抽象”，而是“尽量和 tiny-gpu ISA 对齐”。

代表性 op：

- `tinygpu.add`
- `tinygpu.mul`
- `tinygpu.load`
- `tinygpu.store`
- `tinygpu.shared_load`
- `tinygpu.shared_store`
- `tinygpu.barrier`
- `tinygpu.cmp`
- `tinygpu.branch`
- `tinygpu.jump`
- `tinygpu.ret`

## 优化

文件：

- `include/tiny-gpu-compiler/Passes/Passes.h`
- `lib/Passes/Passes.cpp`

当前优化是直接写在代码里的轻量 pass，不是完整的 MLIR pass pipeline 基础设施重度玩法。

已有 pass：

- Constant Folding
- Dead Code Elimination
- Strength Reduction
- CSE

特征：

- 直接 walk op
- 在本地构造替换 op
- 通过多轮迭代求收敛

## 寄存器分配

文件：

- `lib/CodeGen/RegisterAllocator.cpp`

核心约定：

- `R0-R12`：可分配 GPR
- `R13`：`blockIdx`
- `R14`：`blockDim`
- `R15`：`threadIdx`

这个文件的真正价值在于它把“IR 值”转换成后端能理解的属性契约：

- `rd`
- `rs`
- `rt`

后面的 emitter 就靠这些属性工作。

## 指令发射

文件：

- `lib/CodeGen/TinyGPUEmitter.cpp`
- `include/tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h`

职责：

- 遍历 register-allocated IR
- 为 block 分配 instruction address
- 按 tiny-gpu ISA 编码
- 输出 asm / hex / binary / json trace

最重要的理解点：

### special register 读取不会真的发射指令
例如 `thread_id`, `block_id`, `block_dim` 这些 dialect op，本质上引用的是硬件固定寄存器，因此 emitter 会跳过真正发射。

### branch/jump 依赖 block 地址预计算
emitter 先统计哪些 op 会发射，再把 branch target 编成绝对地址。

## Native 侧你最该记住的一句话

**这个编译器真正的核心不是“把 AST 变成 MLIR”，而是“让 TinyGPU dialect 在带寄存器属性之后，稳定地变成 tiny-gpu ISA”。**
