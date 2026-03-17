#ifndef TGC_PASSES_PASSES_H
#define TGC_PASSES_PASSES_H

#include "mlir/IR/BuiltinOps.h"

namespace tgc {

/// Optimization statistics for reporting
struct OptimizationStats {
  int constantsFolded = 0;
  int deadOpsEliminated = 0;
  int strengthReductions = 0;
  int commonSubexprsEliminated = 0;
  int totalOpsRemoved = 0;

  std::string summary() const;
};

/// Run constant folding: evaluate compile-time known expressions.
/// Example: const 3 + const 5 -> const 8
int runConstantFolding(mlir::Operation *funcOp);

/// Run dead code elimination: remove ops whose results are never used.
int runDeadCodeElimination(mlir::Operation *funcOp);

/// Run strength reduction: replace expensive ops with cheaper alternatives.
/// Example: mul x, 2 -> add x, x; mul x, 1 -> x; mul x, 0 -> const 0
int runStrengthReduction(mlir::Operation *funcOp);

/// Run common subexpression elimination: deduplicate identical computations.
int runCSE(mlir::Operation *funcOp);

/// Run all optimization passes and return statistics.
OptimizationStats runAllOptimizations(mlir::Operation *funcOp);

} // namespace tgc

#endif // TGC_PASSES_PASSES_H
