#ifndef TINYGPU_OPS_H
#define TINYGPU_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.h"

#define GET_OP_CLASSES
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h.inc"

#endif // TINYGPU_OPS_H
