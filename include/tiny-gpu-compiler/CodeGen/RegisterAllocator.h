#ifndef TGC_CODEGEN_REGISTERALLOCATOR_H
#define TGC_CODEGEN_REGISTERALLOCATOR_H

#include "mlir/IR/BuiltinOps.h"

namespace tgc {

/// Assigns physical registers (R0-R12) to SSA values in TinyGPU dialect.
/// Uses a simple linear scan approach. Annotates ops with "reg" attributes.
/// Returns failure if register pressure exceeds 13 GPRs (spilling not yet
/// implemented for the initial version — tiny programs rarely need it).
mlir::LogicalResult allocateRegisters(mlir::Operation *funcOp);

} // namespace tgc

#endif // TGC_CODEGEN_REGISTERALLOCATOR_H
