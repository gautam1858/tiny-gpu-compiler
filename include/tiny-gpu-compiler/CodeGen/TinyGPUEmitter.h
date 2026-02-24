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

struct CompilationTrace {
  std::vector<Instruction> instructions;
  std::string sourceCode;
  std::vector<std::pair<std::string, std::string>> irStages;
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
