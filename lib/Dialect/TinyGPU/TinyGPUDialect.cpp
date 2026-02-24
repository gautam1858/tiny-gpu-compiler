#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"

using namespace mlir;
using namespace mlir::tinygpu;

#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.cpp.inc"

void TinyGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.cpp.inc"
      >();
}
