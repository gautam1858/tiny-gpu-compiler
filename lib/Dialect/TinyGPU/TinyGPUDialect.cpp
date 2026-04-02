#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::tinygpu;

#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.cpp.inc"

void TinyGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.cpp.inc"
      >();
}

Type TinyGPUDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getCurrentLocation(),
                   "tinygpu dialect does not define custom types");
  return Type();
}

void TinyGPUDialect::printType(Type type, DialectAsmPrinter &os) const {
  llvm_unreachable("tinygpu dialect does not define custom types");
}
