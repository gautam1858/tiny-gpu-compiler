#ifndef TGC_CODEGEN_TINYGPUEMITTER_H
#define TGC_CODEGEN_TINYGPUEMITTER_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tgc {

struct Instruction {
  uint16_t binary;
  std::string assembly;
  int address;
};

struct AnalysisResult {
  // Warp divergence analysis
  struct DivergenceInfo {
    int instructionAddr;
    std::string type; // "branch", "converge"
    int divergentThreads;
    int totalThreads;
  };
  std::vector<DivergenceInfo> divergence;

  // Memory coalescing analysis
  struct CoalescingInfo {
    int instructionAddr;
    std::string accessPattern; // "coalesced", "strided", "scattered"
    int transactionsNeeded;
    std::string description;
  };
  std::vector<CoalescingInfo> coalescing;

  // Performance metrics
  int totalInstructions = 0;
  int registersUsed = 0;
  int sharedMemoryBytes = 0;
  int branchInstructions = 0;
  int memoryInstructions = 0;
  int computeInstructions = 0;
  int barrierCount = 0;
  std::string optimizationSummary;
};

struct CompilationTrace {
  std::vector<Instruction> instructions;
  std::string sourceCode;
  std::vector<std::pair<std::string, std::string>> irStages;
  AnalysisResult analysis;
};

/// Emit binary instructions from register-allocated TinyGPU dialect IR.
/// Returns the list of 16-bit instructions.
std::vector<Instruction> emitBinary(mlir::Operation *funcOp);

/// Write binary output as hex text (one instruction per line).
void emitHex(const std::vector<Instruction> &instructions,
             llvm::raw_ostream &os);

/// Write binary output as raw bytes (for simulation loading).
void emitRawBinary(const std::vector<Instruction> &instructions,
                   llvm::raw_ostream &os);

/// Write annotated assembly listing.
void emitAssembly(const std::vector<Instruction> &instructions,
                  llvm::raw_ostream &os);

/// Write JSON trace for the web visualizer.
void emitJsonTrace(const CompilationTrace &trace, llvm::raw_ostream &os);

} // namespace tgc

#endif // TGC_CODEGEN_TINYGPUEMITTER_H
