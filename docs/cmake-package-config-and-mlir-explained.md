# CMake Package Config、`*.cmake` 与 `MLIR_DIR` / `LLVM_DIR` 详解

这篇文档专门解释一个你刚刚碰到、但很多人第一次接触 MLIR / LLVM 项目时都会疑惑的问题：

> `-DMLIR_DIR=...`、`-DLLVM_DIR=...` 到底在指定什么？
> `*.cmake` 是什么？
> 它为什么和我平时只指定编译器、build type 的 CMake 用法不太一样？

我会分成两层来讲：

1. **实用层**：先建立一个足够能干活的心智模型。
2. **深入层**：再把 `find_package(... CONFIG)`、`*Config.cmake`、`*Targets.cmake`、`AddMLIR.cmake` 这些东西的关系真正讲清楚。

文中会尽量用当前 repo 和你机器上的本地 MLIR 安装来举例。

---

## 1. 先说结论：`MLIR_DIR` 不是“中间文件目录”

这是最容易误会的点。

像下面这个参数：

```bash
-DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir"
```

**不是**在指定：

- build 目录
- 中间产物目录
- cache 目录

它指定的是：

> **MLIR 这个外部依赖包的 CMake 配置入口目录**

换句话说，它是在告诉 CMake：

> “如果你要找 MLIR，请从这个目录里的 `MLIRConfig.cmake` 开始找。”

同理：

```bash
-DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm"
```

是在说：

> “如果你要找 LLVM，请从这个目录里的 `LLVMConfig.cmake` 开始找。”

---

## 2. `*.cmake` 到底是什么

从最宽泛的角度说，`*.cmake` 就是 **CMake 脚本文件**。

它不是“一个特殊二进制格式”，本质上就是给 CMake 解释执行的文本脚本。

比如这些都是 `.cmake` 文件：

- `MLIRConfig.cmake`
- `LLVMConfig.cmake`
- `AddMLIR.cmake`
- `HandleLLVMOptions.cmake`
- `FindZLIB.cmake`

它们里面通常会做这些事：

- 定义变量
- 定义函数 / macro
- 导入 target
- 设置 include 路径、库路径
- 帮你把某个外部项目接入当前构建

所以你可以把 `.cmake` 文件理解成：

> **“给 CMake 用的构建说明脚本”**

---

## 3. 为什么你以前不太需要 `XXX_DIR`

因为很多时候你用的依赖都在 **默认搜索路径** 里。

例如系统装的包可能已经把自己的 CMake 配置放进这些地方：

- `/usr/lib/cmake/...`
- `/usr/local/lib/cmake/...`
- `/usr/lib/x86_64-linux-gnu/cmake/...`

此时你写：

```cmake
find_package(OpenCV REQUIRED)
```

它可能就直接找到了，所以你没有感知到“配置入口目录”这件事。

但这次不一样：

- LLVM/MLIR 是你自己装到 `~/llvm-install`
- 这个位置不一定在 CMake 默认搜索前缀里
- 所以 `find_package(MLIR REQUIRED CONFIG)` 不知道去哪里找

这时你就得手动告诉它：

```bash
-DMLIR_DIR=...
-DLLVM_DIR=...
```

---

## 4. 实用层理解：`find_package` 到底在干什么

在这个 repo 里，顶层 `CMakeLists.txt` 有这句：

```cmake
find_package(MLIR REQUIRED CONFIG)
```

这句可以先粗暴理解成：

> “把 MLIR 这套已经安装好的开发环境接进当前项目。”

接进来以后，当前项目通常就能拿到：

- MLIR 的头文件路径
- MLIR 的库 target
- MLIR 提供的 CMake 宏
- MLIR 的一些内部配置变量

而 `CONFIG` 这个关键字说明它走的是：

> **Config package mode**

也就是：

> “去找这个包自己提供的 `MLIRConfig.cmake`。”

这和另一类 `FindXXX.cmake` 机制不同，后面我会讲。

---

## 5. 用这次的 MLIR 例子，实际发生了什么

你本地机器上有这样一个目录：

```bash
$HOME/llvm-install/lib/cmake/mlir
```

里面有这些文件：

- `MLIRConfig.cmake`
- `MLIRConfigVersion.cmake`
- `MLIRTargets.cmake`
- `MLIRTargets-release.cmake`
- `AddMLIR.cmake`
- `AddMLIRPython.cmake`

以及另一个 LLVM 目录：

```bash
$HOME/llvm-install/lib/cmake/llvm
```

里面有：

- `LLVMConfig.cmake`
- `LLVMConfigVersion.cmake`
- `LLVMExports.cmake`
- `LLVMExports-release.cmake`
- `AddLLVM.cmake`
- `HandleLLVMOptions.cmake`
- `TableGen.cmake`

所以这次 configure 的时候：

```bash
cmake -G Ninja -S . -B build \
  -DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir" \
  -DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm"
```

基本相当于在告诉 CMake：

- MLIR 配置入口在这里
- LLVM 配置入口在这里

然后 `find_package(MLIR REQUIRED CONFIG)` 就能成功。

---

## 6. 当前 repo 是怎样用这些配置的

当前 repo 顶层 `CMakeLists.txt` 里有：

```cmake
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
```

这几行非常关键。

它表达的是一个典型的 LLVM/MLIR 项目接入方式：

1. 先用 `find_package(MLIR CONFIG)` 找到 MLIR 包配置。
2. 找到之后，MLIR/LLVM 会暴露出一些变量，例如：
   - `MLIR_CMAKE_DIR`
   - `LLVM_CMAKE_DIR`
   - `MLIR_INCLUDE_DIRS`
   - `LLVM_INCLUDE_DIRS`
3. 然后项目把这些目录追加进 `CMAKE_MODULE_PATH`。
4. 于是当前项目就能 `include(AddMLIR)`、`include(AddLLVM)`、`include(TableGen)`。

也就是说：

> `find_package(MLIR CONFIG)` 只是第一步，
> 它的一个重要作用是把后续这些 MLIR / LLVM 提供的构建辅助模块暴露出来。

---

## 7. `MLIRConfig.cmake` 一般会包含哪些信息

可以把它理解成：

> **“MLIR 安装包对外暴露的入口脚本”**

它通常会做下面几类事：

### 7.1 暴露包版本和基础变量

例如：

- 当前 MLIR 安装根位置
- include 目录
- library 目录
- 配套 CMake 模块目录

常见变量风格大概像这样：

- `MLIR_INCLUDE_DIRS`
- `MLIR_CMAKE_DIR`
- `MLIR_TABLEGEN_EXE`

### 7.2 引入目标定义

它一般不会把所有 target 都直接写死在自己里面，而是继续 include：

- `MLIRTargets.cmake`
- 或者相关 export 文件

这些文件负责定义 imported targets。

### 7.3 引出依赖关系

MLIR 是建在 LLVM 上面的，所以 `MLIRConfig.cmake` 通常也会和 LLVM 配置产生联系，最终让你能同时接入 LLVM 和 MLIR 的构建信息。

### 7.4 提供后续模块入口

比如让你后续能 include：

- `AddMLIR.cmake`
- `AddLLVM.cmake`
- `TableGen.cmake`

---

## 8. `MLIRTargets.cmake` 又是干什么的

如果说 `MLIRConfig.cmake` 是入口，那 `MLIRTargets.cmake` 更像：

> **“这个包里有哪些可供链接和引用的 target 清单”**

它主要负责导入这些东西：

- `MLIRIR`
- `MLIRSupport`
- `MLIRParser`
- 以及其他 MLIR 组件 target

这些通常是 **imported targets**。

你可以把 imported target 理解成：

> “这个 target 不是当前工程自己 `add_library()` 出来的，而是从外部已安装包导进来的。”

也就是说，当你在自己的 CMake 里写：

```cmake
target_link_libraries(mytool PRIVATE MLIRIR)
```

前提就是某个 `*Targets.cmake` 已经把 `MLIRIR` 定义好了。

---

## 9. `AddMLIR.cmake` 和 `AddLLVM.cmake` 是什么

这两个文件不是“库目标清单”，而是：

> **辅助构建宏 / function 的集合**

例如你在 LLVM/MLIR 项目里常见到：

- `add_mlir_library(...)`
- `add_mlir_dialect(...)`
- `mlir_tablegen(...)`
- `add_llvm_executable(...)`

这些命令并不是 CMake 原生自带的。

它们之所以能用，是因为项目里先：

```cmake
include(AddMLIR)
include(AddLLVM)
```

这些模块被 include 进来之后，宏和函数就可用了。

所以它们的角色更像：

- `MLIRConfig.cmake`：入口
- `MLIRTargets.cmake`：目标清单
- `AddMLIR.cmake`：构建工具箱

---

## 10. `CONFIG` 模式和 `FindXXX.cmake` 模式有什么区别

这是 CMake 里一个很重要但经常被忽略的区别。

### 10.1 Config mode

例如：

```cmake
find_package(MLIR REQUIRED CONFIG)
```

意思是：

> “优先去找 MLIR 自己安装时导出的 `MLIRConfig.cmake`。”

这是**包作者自己提供**的官方接入方式。

优点：

- 最准确
- 目标定义最完整
- 和包本身版本最匹配

### 10.2 Module mode

例如：

```cmake
find_package(ZLIB REQUIRED)
```

如果没写 `CONFIG`，CMake 可能会去找：

- `FindZLIB.cmake`

这类文件通常是 **CMake 自己** 或系统提供的“查找脚本”。

它不是包作者导出的，而是第三方写的“如何猜出这个库装在哪、头文件在哪、库在哪”。

### 10.3 为什么 MLIR/LLVM 更适合 CONFIG

因为 LLVM/MLIR 很复杂：

- target 多
- 宏多
- TableGen 工具链多
- 版本耦合强

用 `Config.cmake` 比让外部 `FindMLIR.cmake` 来猜稳定得多。

---

## 11. CMake 是怎么找到 `MLIRConfig.cmake` 的

这里讲查找链路。

当你写：

```cmake
find_package(MLIR REQUIRED CONFIG)
```

它会尝试在一系列位置找：

- `MLIR_DIR` 指定的位置
- `CMAKE_PREFIX_PATH` 下的标准子目录
- 系统默认安装前缀
- 一些平台默认路径

### 11.1 如果你显式传了 `MLIR_DIR`

例如：

```bash
-DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir"
```

那几乎就是告诉它：

> “别猜了，就去这里找。”

这是最直接、最稳定的方式。

### 11.2 如果你不传 `MLIR_DIR`，可能也能靠 `CMAKE_PREFIX_PATH`

例如：

```bash
-DCMAKE_PREFIX_PATH="$HOME/llvm-install"
```

CMake 就会尝试在这个 prefix 下找典型子目录，比如：

- `lib/cmake/mlir`
- `lib/cmake/llvm`

所以很多场景里，下面两种写法都可能成立：

```bash
-DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir"
-DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm"
```

或者：

```bash
-DCMAKE_PREFIX_PATH="$HOME/llvm-install"
```

但在 LLVM/MLIR 这类版本复杂的场景里，显式 `MLIR_DIR` / `LLVM_DIR` 更少歧义。

---

## 12. 为什么不能只靠 PATH

这也是很常见的误区。

`PATH` 主要是给 shell 和 CMake 找 **可执行文件** 用的。

比如：

- `clang`
- `clang++`
- `mlir-opt`
- `FileCheck`

但 `find_package(MLIR CONFIG)` 找的不是可执行文件，而是：

- `MLIRConfig.cmake`

这类文件不靠 `PATH` 搜索。

所以出现下面这种情况完全正常：

- 你能运行 `mlir-opt`
- 但 CMake 还是报 `Could not find MLIRConfig.cmake`

因为这是两套不同的查找机制。

---

## 13. 把这件事类比成你自己写的库

假设你写了一个自己的库 `MyMath`，并且把它安装成一个标准 CMake package。

安装后你可能会有：

```bash
/some/install/prefix/lib/cmake/MyMath/
  MyMathConfig.cmake
  MyMathConfigVersion.cmake
  MyMathTargets.cmake
```

别人要用你的库时就可以写：

```cmake
find_package(MyMath REQUIRED CONFIG)
target_link_libraries(app PRIVATE MyMath::MyMath)
```

如果你的安装前缀不在默认路径，别人就得传：

```bash
-DMyMath_DIR=/some/install/prefix/lib/cmake/MyMath
```

这和 `MLIR_DIR` 是同一个套路，只是 MLIR 的规模大得多。

---

## 14. 回到这次 tiny-gpu-compiler 的具体情境

这次你之所以必须显式传：

```bash
-DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir"
-DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm"
```

核心原因有三个：

### 14.1 依赖不是系统默认位置

LLVM/MLIR 装在 `~/llvm-install`，不是 `/usr` 或 `/usr/local`。

### 14.2 这个项目不是只“链接一个库”

它不仅需要头文件和库，还需要：

- `AddMLIR.cmake`
- `AddLLVM.cmake`
- `TableGen.cmake`
- `HandleLLVMOptions.cmake`

也就是需要整套 LLVM/MLIR 的构建辅助工具链。

### 14.3 版本必须精确匹配

这个 repo 当前是按 LLVM/MLIR **18.1.8** 跑通的。

如果 CMake 不小心找到系统里别的 LLVM/MLIR 版本，可能会出现：

- 头文件不匹配
- 宏不匹配
- ODS / TableGen 规则不匹配
- 链接目标不匹配

所以显式指定目录更稳。

---

## 15. 你可以怎么检查一个包配置到底有没有被正确找到

最直接的方法：

### 15.1 看配置命令里显式指定的目录

```bash
-DMLIR_DIR="$HOME/llvm-install/lib/cmake/mlir"
-DLLVM_DIR="$HOME/llvm-install/lib/cmake/llvm"
```

### 15.2 看项目自己的 CMake 输出

当前 repo 里有：

```cmake
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
```

所以 configure 时会打印它实际使用了哪个目录。

### 15.3 直接看目录里有没有配置入口文件

```bash
ls "$HOME/llvm-install/lib/cmake/mlir"
ls "$HOME/llvm-install/lib/cmake/llvm"
```

你应该能看到：

- `MLIRConfig.cmake`
- `LLVMConfig.cmake`

---

## 16. 一个最实用的心智模型

如果你不想记太多术语，可以先只记这套：

### 16.1 `CMAKE_CXX_COMPILER`

是在说：

> “拿哪个编译器来编我自己的代码”

### 16.2 `CMAKE_BUILD_TYPE`

是在说：

> “用什么优化/调试模式构建”

### 16.3 `MLIR_DIR` / `LLVM_DIR`

是在说：

> “我依赖的外部工具链，它们的 CMake 配置入口在哪”

### 16.4 `find_package(MLIR CONFIG)`

是在说：

> “去读 MLIR 自己导出的 CMake 接入说明书，然后把它接进当前工程”

### 16.5 `include(AddMLIR)`

是在说：

> “把 MLIR 附带的 CMake 构建工具箱加载进来”

---

## 17. 一个常见误区总结

### 误区 1：`MLIR_DIR` 是 build 目录

不是。

它是 **包配置目录**。

### 误区 2：我能运行 `mlir-opt`，CMake 就一定能找到 MLIR

不是。

`mlir-opt` 走的是可执行文件查找；`find_package` 找的是 `MLIRConfig.cmake`。

### 误区 3：`*.cmake` 就等于一个“包”

不完全对。

更准确地说：

- `.cmake` 是 CMake 脚本文件
- 其中某些特定文件（如 `MLIRConfig.cmake`）扮演“包配置入口”的角色

### 误区 4：`MLIR_DIR` 和 `CMAKE_MODULE_PATH` 是一回事

不是。

- `MLIR_DIR` 是给 `find_package(... CONFIG)` 找配置入口用的
- `CMAKE_MODULE_PATH` 是给 `include(...)` / module 查找用的

在这个 repo 里，流程是：

1. 先通过 `MLIR_DIR` 找到 MLIR
2. 再用 MLIR 提供的 `MLIR_CMAKE_DIR` 去扩展 `CMAKE_MODULE_PATH`
3. 然后才能 `include(AddMLIR)`

---

## 18. 一句话版总结

如果你以后再看到：

```bash
-DXXX_DIR=/some/path/lib/cmake/XXX
```

优先把它理解成：

> **“这是在告诉 CMake：外部依赖 XXX 的 `XXXConfig.cmake` 在哪里。”**

而不是：

> “这是在指定编译输出或中间文件位置。”

---

## 19. 对当前 repo 的一句最贴切解释

当前 `tiny-gpu-compiler` 之所以要显式传 `MLIR_DIR` / `LLVM_DIR`，是因为它不是一个“普通只链接库”的项目，而是一个依赖 **LLVM/MLIR 整套 CMake 构建生态** 的项目：

- 要找到 LLVM / MLIR 的库
- 要找到 LLVM / MLIR 的头文件
- 要找到 `AddMLIR.cmake` / `AddLLVM.cmake` / `TableGen.cmake`
- 要用它们的构建宏和生成流程

所以它需要的不只是“某个 `.so` 文件在哪”，而是：

> **“这整套 LLVM/MLIR 的 CMake 入口在哪。”**

这就是 `MLIR_DIR` / `LLVM_DIR` 的真正意义。
