#include "tiny-gpu-compiler/CodeGen/RegisterAllocator.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace tgc {

/// Simple linear scan register allocator for tiny-gpu's 13 GPRs (R0-R12).
/// R13=%blockIdx, R14=%blockDim, R15=%threadIdx are reserved for hardware.
///
/// Strategy: Walk the function in order, assign registers to each SSA value
/// that produces a result. Since tiny-gpu programs are small (typically <30
/// instructions), we use a greedy approach: assign the lowest available register,
/// and free registers when their last use is encountered.
LogicalResult allocateRegisters(Operation *op) {
  constexpr int NUM_GPRS = 13; // R0-R12
  constexpr int REG_BLOCK_IDX = 13;
  constexpr int REG_BLOCK_DIM = 14;
  constexpr int REG_THREAD_IDX = 15;

  // Collect all operations in order
  std::vector<Operation *> ops;
  op->walk([&](Operation *child) {
    if (child != op)
      ops.push_back(child);
  });

  // Build last-use map: for each Value, find the index of its last use
  llvm::DenseMap<Value, int> lastUse;
  for (int i = 0; i < (int)ops.size(); i++) {
    for (Value operand : ops[i]->getOperands()) {
      lastUse[operand] = i;
    }
  }

  // Track register assignments
  llvm::DenseMap<Value, int> regMap;
  std::vector<bool> regInUse(NUM_GPRS, false);

  auto allocReg = [&]() -> int {
    for (int r = 0; r < NUM_GPRS; r++) {
      if (!regInUse[r]) {
        regInUse[r] = true;
        return r;
      }
    }
    return -1; // No register available
  };

  auto freeReg = [&](int r) {
    if (r >= 0 && r < NUM_GPRS)
      regInUse[r] = false;
  };

  for (int i = 0; i < (int)ops.size(); i++) {
    Operation *currOp = ops[i];

    // Assign registers to results
    for (Value result : currOp->getResults()) {
      int reg = -1;

      // Special register ops get fixed assignments
      if (isa<tinygpu::BlockIdOp>(currOp)) {
        reg = REG_BLOCK_IDX;
      } else if (isa<tinygpu::BlockDimOp>(currOp)) {
        reg = REG_BLOCK_DIM;
      } else if (isa<tinygpu::ThreadIdOp>(currOp)) {
        reg = REG_THREAD_IDX;
      } else {
        reg = allocReg();
        if (reg < 0) {
          currOp->emitError("register allocation failed: all 13 GPRs in use");
          return failure();
        }
      }

      regMap[result] = reg;
      currOp->setAttr("rd", IntegerAttr::get(
                                IntegerType::get(currOp->getContext(), 32), reg));
    }

    // Annotate operand registers
    for (int opIdx = 0; opIdx < (int)currOp->getNumOperands(); opIdx++) {
      Value operand = currOp->getOperand(opIdx);
      auto it = regMap.find(operand);
      if (it != regMap.end()) {
        std::string attrName = (opIdx == 0) ? "rs" : "rt";
        currOp->setAttr(attrName,
                        IntegerAttr::get(
                            IntegerType::get(currOp->getContext(), 32),
                            it->second));
      }
    }

    // Free registers whose values are no longer needed
    for (Value operand : currOp->getOperands()) {
      auto useIt = lastUse.find(operand);
      if (useIt != lastUse.end() && useIt->second == i) {
        auto regIt = regMap.find(operand);
        if (regIt != regMap.end() && regIt->second < NUM_GPRS) {
          freeReg(regIt->second);
        }
      }
    }
  }

  return success();
}

} // namespace tgc
