#include "tiny-gpu-compiler/Passes/Passes.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <sstream>

using namespace mlir;

namespace tgc {

std::string OptimizationStats::summary() const {
  std::ostringstream ss;
  ss << "Optimizations: ";
  if (constantsFolded > 0)
    ss << constantsFolded << " constants folded, ";
  if (deadOpsEliminated > 0)
    ss << deadOpsEliminated << " dead ops eliminated, ";
  if (strengthReductions > 0)
    ss << strengthReductions << " strength reductions, ";
  if (commonSubexprsEliminated > 0)
    ss << commonSubexprsEliminated << " CSE eliminations, ";
  if (totalOpsRemoved == 0)
    ss << "no optimizations applied";
  else
    ss << totalOpsRemoved << " total ops removed";
  return ss.str();
}

/// Check if an operation is a tinygpu.const with a given value
static bool isConstWithValue(Operation *op, uint8_t &val) {
  if (auto constOp = dyn_cast<tinygpu::ConstOp>(op)) {
    val = constOp.getValue();
    return true;
  }
  return false;
}

int runConstantFolding(Operation *funcOp) {
  int folded = 0;
  llvm::SmallVector<Operation *, 16> toErase;

  funcOp->walk([&](Operation *op) {
    uint8_t lhsVal, rhsVal;

    if (auto addOp = dyn_cast<tinygpu::AddOp>(op)) {
      if (addOp.getLhs().getDefiningOp() &&
          isConstWithValue(addOp.getLhs().getDefiningOp(), lhsVal) &&
          addOp.getRhs().getDefiningOp() &&
          isConstWithValue(addOp.getRhs().getDefiningOp(), rhsVal)) {
        OpBuilder builder(op);
        auto i8Ty = builder.getI8Type();
        auto result = builder.create<tinygpu::ConstOp>(
            op->getLoc(), i8Ty, (uint8_t)((lhsVal + rhsVal) & 0xFF));
        op->getResult(0).replaceAllUsesWith(result);
        toErase.push_back(op);
        folded++;
      }
    } else if (auto subOp = dyn_cast<tinygpu::SubOp>(op)) {
      if (subOp.getLhs().getDefiningOp() &&
          isConstWithValue(subOp.getLhs().getDefiningOp(), lhsVal) &&
          subOp.getRhs().getDefiningOp() &&
          isConstWithValue(subOp.getRhs().getDefiningOp(), rhsVal)) {
        OpBuilder builder(op);
        auto i8Ty = builder.getI8Type();
        auto result = builder.create<tinygpu::ConstOp>(
            op->getLoc(), i8Ty, (uint8_t)((lhsVal - rhsVal) & 0xFF));
        op->getResult(0).replaceAllUsesWith(result);
        toErase.push_back(op);
        folded++;
      }
    } else if (auto mulOp = dyn_cast<tinygpu::MulOp>(op)) {
      if (mulOp.getLhs().getDefiningOp() &&
          isConstWithValue(mulOp.getLhs().getDefiningOp(), lhsVal) &&
          mulOp.getRhs().getDefiningOp() &&
          isConstWithValue(mulOp.getRhs().getDefiningOp(), rhsVal)) {
        OpBuilder builder(op);
        auto i8Ty = builder.getI8Type();
        auto result = builder.create<tinygpu::ConstOp>(
            op->getLoc(), i8Ty, (uint8_t)((lhsVal * rhsVal) & 0xFF));
        op->getResult(0).replaceAllUsesWith(result);
        toErase.push_back(op);
        folded++;
      }
    }
  });

  for (auto *op : toErase)
    op->erase();

  return folded;
}

int runDeadCodeElimination(Operation *funcOp) {
  int eliminated = 0;
  bool changed = true;

  while (changed) {
    changed = false;
    llvm::SmallVector<Operation *, 16> toErase;

    funcOp->walk([&](Operation *op) {
      // Skip terminators and side-effecting ops
      if (op->hasTrait<OpTrait::IsTerminator>())
        return;
      if (isa<tinygpu::StoreOp, tinygpu::SharedStoreOp, tinygpu::BarrierOp,
              tinygpu::FuncOp>(op))
        return;

      // If the op has results and none are used, it's dead
      if (op->getNumResults() > 0) {
        bool allResultsUnused = true;
        for (Value result : op->getResults()) {
          if (!result.use_empty()) {
            allResultsUnused = false;
            break;
          }
        }
        if (allResultsUnused) {
          toErase.push_back(op);
          eliminated++;
          changed = true;
        }
      }
    });

    for (auto *op : toErase)
      op->erase();
  }

  return eliminated;
}

int runStrengthReduction(Operation *funcOp) {
  int reduced = 0;
  llvm::SmallVector<Operation *, 16> toErase;

  funcOp->walk([&](Operation *op) {
    if (auto mulOp = dyn_cast<tinygpu::MulOp>(op)) {
      uint8_t val;
      // mul x, 0 -> const 0
      if (mulOp.getRhs().getDefiningOp() &&
          isConstWithValue(mulOp.getRhs().getDefiningOp(), val) && val == 0) {
        OpBuilder builder(op);
        auto i8Ty = builder.getI8Type();
        auto zero = builder.create<tinygpu::ConstOp>(op->getLoc(), i8Ty,
                                                     (uint8_t)0);
        op->getResult(0).replaceAllUsesWith(zero);
        toErase.push_back(op);
        reduced++;
        return;
      }
      // mul x, 1 -> x
      if (mulOp.getRhs().getDefiningOp() &&
          isConstWithValue(mulOp.getRhs().getDefiningOp(), val) && val == 1) {
        op->getResult(0).replaceAllUsesWith(mulOp.getLhs());
        toErase.push_back(op);
        reduced++;
        return;
      }
      // mul x, 2 -> add x, x
      if (mulOp.getRhs().getDefiningOp() &&
          isConstWithValue(mulOp.getRhs().getDefiningOp(), val) && val == 2) {
        OpBuilder builder(op);
        auto i8Ty = builder.getI8Type();
        auto add = builder.create<tinygpu::AddOp>(op->getLoc(), i8Ty,
                                                  mulOp.getLhs(), mulOp.getLhs());
        op->getResult(0).replaceAllUsesWith(add);
        toErase.push_back(op);
        reduced++;
        return;
      }
      // Same for lhs
      if (mulOp.getLhs().getDefiningOp() &&
          isConstWithValue(mulOp.getLhs().getDefiningOp(), val) && val == 0) {
        OpBuilder builder(op);
        auto i8Ty = builder.getI8Type();
        auto zero = builder.create<tinygpu::ConstOp>(op->getLoc(), i8Ty,
                                                     (uint8_t)0);
        op->getResult(0).replaceAllUsesWith(zero);
        toErase.push_back(op);
        reduced++;
        return;
      }
      if (mulOp.getLhs().getDefiningOp() &&
          isConstWithValue(mulOp.getLhs().getDefiningOp(), val) && val == 1) {
        op->getResult(0).replaceAllUsesWith(mulOp.getRhs());
        toErase.push_back(op);
        reduced++;
        return;
      }
    }

    // add x, 0 -> x
    if (auto addOp = dyn_cast<tinygpu::AddOp>(op)) {
      uint8_t val;
      if (addOp.getRhs().getDefiningOp() &&
          isConstWithValue(addOp.getRhs().getDefiningOp(), val) && val == 0) {
        op->getResult(0).replaceAllUsesWith(addOp.getLhs());
        toErase.push_back(op);
        reduced++;
        return;
      }
      if (addOp.getLhs().getDefiningOp() &&
          isConstWithValue(addOp.getLhs().getDefiningOp(), val) && val == 0) {
        op->getResult(0).replaceAllUsesWith(addOp.getRhs());
        toErase.push_back(op);
        reduced++;
        return;
      }
    }

    // sub x, 0 -> x
    if (auto subOp = dyn_cast<tinygpu::SubOp>(op)) {
      uint8_t val;
      if (subOp.getRhs().getDefiningOp() &&
          isConstWithValue(subOp.getRhs().getDefiningOp(), val) && val == 0) {
        op->getResult(0).replaceAllUsesWith(subOp.getLhs());
        toErase.push_back(op);
        reduced++;
        return;
      }
    }
  });

  for (auto *op : toErase)
    op->erase();

  return reduced;
}

int runCSE(Operation *funcOp) {
  int eliminated = 0;

  // Build a map from (opcode, operand1, operand2) -> existing result
  // Only for pure ops (arithmetic, const)
  llvm::SmallVector<Operation *, 16> toErase;

  for (Block &block : funcOp->getRegion(0)) {
    llvm::DenseMap<std::pair<const void *, std::pair<Value, Value>>, Value>
        binaryCSE;
    llvm::DenseMap<uint8_t, Value> constCSE;

    for (Operation &op : block) {
      // CSE for const ops
      if (auto constOp = dyn_cast<tinygpu::ConstOp>(&op)) {
        uint8_t val = constOp.getValue();
        auto it = constCSE.find(val);
        if (it != constCSE.end()) {
          op.getResult(0).replaceAllUsesWith(it->second);
          toErase.push_back(&op);
          eliminated++;
          continue;
        }
        constCSE[val] = op.getResult(0);
      }

      // CSE for binary ops
      if (isa<tinygpu::AddOp, tinygpu::SubOp, tinygpu::MulOp,
              tinygpu::DivOp>(&op) &&
          op.getNumOperands() == 2) {
        auto key = std::make_pair(
            op.getName().getAsOpaquePointer(),
            std::make_pair(op.getOperand(0), op.getOperand(1)));
        auto it = binaryCSE.find(key);
        if (it != binaryCSE.end()) {
          op.getResult(0).replaceAllUsesWith(it->second);
          toErase.push_back(&op);
          eliminated++;
          continue;
        }
        binaryCSE[key] = op.getResult(0);
      }
    }
  }

  for (auto *op : toErase)
    op->erase();

  return eliminated;
}

OptimizationStats runAllOptimizations(Operation *funcOp) {
  OptimizationStats stats;

  // Run passes in order, multiple iterations for convergence
  for (int iter = 0; iter < 3; iter++) {
    stats.constantsFolded += runConstantFolding(funcOp);
    stats.strengthReductions += runStrengthReduction(funcOp);
    stats.commonSubexprsEliminated += runCSE(funcOp);
    stats.deadOpsEliminated += runDeadCodeElimination(funcOp);
  }

  stats.totalOpsRemoved = stats.constantsFolded + stats.deadOpsEliminated +
                           stats.strengthReductions +
                           stats.commonSubexprsEliminated;
  return stats;
}

} // namespace tgc
