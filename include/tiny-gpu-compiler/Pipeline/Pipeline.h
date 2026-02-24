#ifndef TGC_PIPELINE_PIPELINE_H
#define TGC_PIPELINE_PIPELINE_H

#include "tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace tgc {

enum class OutputFormat {
  MLIR,       // Dump the TinyGPU dialect MLIR
  Assembly,   // Human-readable annotated assembly
  Hex,        // Hex text (one instruction per line)
  Binary,     // Raw binary
  JsonTrace,  // Full compilation trace for visualizer
};

struct CompilerOptions {
  OutputFormat format = OutputFormat::Assembly;
  bool dumpAfterEachPass = false;
};

/// Run the full compilation pipeline: parse .tgc source, generate MLIR,
/// allocate registers, and emit binary.
/// Returns the compilation trace (populated if format == JsonTrace).
CompilationTrace compile(const std::string &source,
                         const CompilerOptions &opts,
                         llvm::raw_ostream &os);

} // namespace tgc

#endif // TGC_PIPELINE_PIPELINE_H
