# Web 端架构与 Native 的关系

## 先说结论

web 端不是 native 编译器的前端壳子。

它是：

- 一套 React UI
- 一套浏览器内 TypeScript 编译器
- 一套浏览器内 GPU simulator

也就是说，这是 repo 里的**第二套编译实现**。

## 入口

- `web/src/main.tsx`
- `web/src/App.tsx`

`App.tsx` 负责把编辑器、pipeline view、binary view、simulator、analysis panel 拼起来。

## 核心文件

### 浏览器编译器
- `web/src/compiler/TGCCompiler.ts`

### 类型定义
- `web/src/compiler/types.ts`

### 模拟器
- `web/src/simulator/TinyGPUSim.ts`

### 组件
- `web/src/components/Editor.tsx`
- `web/src/components/PipelineView.tsx`
- `web/src/components/BinaryView.tsx`
- `web/src/components/GPUSimulator.tsx`
- `web/src/components/AnalysisPanel.tsx`

## 它和 C++ 侧相似的地方

相似点：

- 都有 lexer/parser/compiler pipeline 的概念
- 都有“IR stages”
- 都会生成 instruction 列表
- 都会做分析和模拟展示

这也是为什么 UI 看起来像“直接展示 native pipeline”。

## 它和 C++ 侧不同的地方

这部分最容易让新人误判。

### 1. 它不使用 MLIR
`web/src/compiler/TGCCompiler.ts` 是自己维护 AST/编译逻辑的，不是绑定到 C++/MLIR 编译器。

### 2. register allocation 明显不同
C++ 侧是线性扫描式思路；web 侧更偏简单映射和局部复用。

所以：

- register pressure
- reuse 行为
- failure mode

都不应默认和 native 一致。

### 3. optimization 不等价
C++ 有真实 pass：

- constant folding
- strength reduction
- CSE
- DCE

web 侧更多是为了教学展示，不能把它当成和 native 完全等价的 optimizer。

### 4. analysis 数据来源不同
web analysis panel 更多依赖 TypeScript 那套 trace/analysis 结构，而不是完整复用 native 编译器的分析输出。

## Simulator

文件：

- `web/src/simulator/TinyGPUSim.ts`

这里模拟：

- program memory
- data memory
- shared memory
- threads
- block dispatch
- barrier
- divergence

它的设计目标是“足够说明 GPU 执行模型”，不是做工业级精确仿真。

## UI 中最值得读的组件

### `PipelineView.tsx`
适合理解 UI 如何把 source + stages 展示为可点击阶段。

### `GPUSimulator.tsx`
适合理解用户如何 step/run，以及 simulator state 怎样被消费。

### `AnalysisPanel.tsx`
适合理解这个项目打算教给用户哪些 GPU 性能概念。

## 对新人最重要的提醒

调试时先确认你在看哪一套系统：

- 如果你在看正确性、dialect、emitter、寄存器分配：去 C++
- 如果你在看可视化、交互、浏览器模拟体验：去 web
- 不要默认 web 编译结果与 native 逐字逐寄存器一致
