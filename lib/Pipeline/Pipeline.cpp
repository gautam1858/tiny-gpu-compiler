#include "tiny-gpu-compiler/Pipeline/Pipeline.h"
#include "tiny-gpu-compiler/CodeGen/RegisterAllocator.h"
#include "tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"
#include "tiny-gpu-compiler/Frontend/Lexer.h"
#include "tiny-gpu-compiler/Frontend/MLIRGen.h"
#include "tiny-gpu-compiler/Frontend/Parser.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace tgc {

/// Capture the current module IR as a string.
static std::string captureIR(ModuleOp module) {
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream);
  return ir;
}

CompilationTrace compile(const std::string &source,
                         const CompilerOptions &opts,
                         llvm::raw_ostream &os) {
  CompilationTrace trace;
  trace.sourceCode = source;

  // Set up MLIR context with our dialect
  MLIRContext context;
  context.getOrLoadDialect<tinygpu::TinyGPUDialect>();

  // Stage 1: Frontend — Parse .tgc source to AST
  Lexer lexer(source);
  Parser parser(lexer);
  auto program = parser.parseProgram();

  // Stage 2: MLIRGen — AST to MLIR (TinyGPU dialect)
  auto module = mlirGen(context, *program);
  if (!module) {
    llvm::errs() << "MLIRGen failed\n";
    return trace;
  }

  trace.irStages.push_back({"Frontend → TinyGPU Dialect", captureIR(*module)});

  if (opts.format == OutputFormat::MLIR) {
    module->print(os);
    return trace;
  }

  // Stage 3: Register Allocation
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      if (failed(allocateRegisters(&op))) {
        llvm::errs() << "Register allocation failed\n";
        return trace;
      }
    }
  }

  trace.irStages.push_back(
      {"Register Allocation", captureIR(*module)});

  // Stage 4: Binary Emission
  std::vector<Instruction> allInstructions;
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      auto instructions = emitBinary(&op);
      allInstructions.insert(allInstructions.end(), instructions.begin(),
                             instructions.end());
    }
  }

  trace.instructions = allInstructions;

  switch (opts.format) {
  case OutputFormat::Assembly:
    emitAssembly(allInstructions, os);
    break;
  case OutputFormat::Hex:
    emitHex(allInstructions, os);
    break;
  case OutputFormat::Binary:
    emitRawBinary(allInstructions, os);
    break;
  case OutputFormat::JsonTrace:
    emitJsonTrace(trace, os);
    break;
  default:
    break;
  }

  return trace;
}

} // namespace tgc
