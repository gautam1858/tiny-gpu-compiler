#ifndef TGC_FRONTEND_MLIRGEN_H
#define TGC_FRONTEND_MLIRGEN_H

#include "tiny-gpu-compiler/Frontend/AST.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace tgc {

/// Generate MLIR from a parsed TGC program.
/// Produces gpu + arith + memref + scf dialect operations.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                           const Program &program);

} // namespace tgc

#endif // TGC_FRONTEND_MLIRGEN_H
