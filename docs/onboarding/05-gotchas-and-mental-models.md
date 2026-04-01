# 容易踩坑的地方

## 1. 这个 repo 不是“一个编译器 + 一个 UI”

更准确地说，是：

- 一个 native C++/MLIR 编译器
- 一个 browser-side TypeScript 编译器/模拟器

这会导致很多“看起来应该一样”的地方其实不一样。

## 2. 参数传递不是函数参数语义

`tinygpu.func` 没有显式参数。

参数是固定地址协议：

- global pointers 映射到固定 base
- scalar int 从高地址区加载

如果你带着普通编译器的 calling convention 心智模型来看，会很容易误读代码。

## 3. shared memory 是隐式布局，不是动态资源系统

shared arrays 只是按顺序分配 offset。

含义：

- 没有复杂 allocator
- 没有强边界保护
- 你加新特性时要自己关注 64-byte shared memory 上限

## 4. special registers 不会真的生成指令

`blockIdx` / `blockDim` / `threadIdx` 对应固定寄存器。

所以你在 dialect 里看到读这些值的 op，不代表 emitter 会生成一条实际 instruction。

## 5. `rd/rs/rt` 才是后端真正的桥

如果你只看 dialect 定义，会以为“op 已经足够接近硬件了”。

其实真正让 emitter 能工作的是 register allocator 写下的属性：

- `rd`
- `rs`
- `rt`

忘了这层，就会误解后端实现。

## 6. condition lowering 需要谨慎理解

比较与分支路径里，`cmp` / NZP / branch mask 的语义需要仔细核对，不能想当然把它当成普通 C 布尔值流转。

如果你在调：

- `if`
- `for`
- comparison

优先读：

- `lib/Frontend/MLIRGen.cpp`
- `include/.../TinyGPUOps.td`
- `lib/CodeGen/TinyGPUEmitter.cpp`

## 7. web analysis 不等于 native analysis

analysis panel 展示的是项目教学视角下的分析结果，但来源和 native trace/analysis 并不是完全同一套管线。

因此遇到差异时，不要先怀疑 UI，先确认分析数据来自哪一侧。

## 8. 这个项目更像“可解释编译器”而不是“追求工业完备性”

很多设计是故意简化的：

- 8-bit 数据通路
- 固定内存布局
- 简化寄存器分配
- 小而透明的 pass 集合
- 直观的 ISA 对应关系

这不是缺点，而是项目目标的一部分。

## 建议你记住的三个心智模型

### 心智模型 A：两套系统
- native 负责“真实编译器语义”
- web 负责“教学与交互复现”

### 心智模型 B：一条核心契约链
DSL → AST → TinyGPU dialect → `rd/rs/rt` → ISA

### 心智模型 C：内存/寄存器协议优先
这个项目很多“语义”其实最终都落到固定硬件协议上：

- 参数 base 地址
- shared memory offset
- reserved registers
- branch target address
