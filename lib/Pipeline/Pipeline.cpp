#include "tiny-gpu-compiler/Pipeline/Pipeline.h"
#include "tiny-gpu-compiler/CodeGen/RegisterAllocator.h"
#include "tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"
#include "tiny-gpu-compiler/Frontend/Lexer.h"
#include "tiny-gpu-compiler/Frontend/MLIRGen.h"
#include "tiny-gpu-compiler/Frontend/Parser.h"
#include "tiny-gpu-compiler/Passes/Passes.h"

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

/// Analyze the compiled instructions for divergence and coalescing patterns.
static void analyzeInstructions(CompilationTrace &trace) {
  auto &analysis = trace.analysis;
  analysis.totalInstructions = trace.instructions.size();

  for (auto &inst : trace.instructions) {
    uint16_t binary = inst.binary;
    uint16_t opcode = (binary >> 12) & 0xF;

    switch (opcode) {
    case 0b0001: // BRnzp
      analysis.branchInstructions++;
      {
        AnalysisResult::DivergenceInfo div;
        div.instructionAddr = inst.address;
        div.type = "branch";
        div.divergentThreads = 0; // static estimate
        div.totalThreads = 0;
        analysis.divergence.push_back(div);
      }
      break;
    case 0b0111: // LDR
    case 0b1000: // STR
    case 0b1010: // SLDR
    case 0b1011: // SSTR
      analysis.memoryInstructions++;
      {
        AnalysisResult::CoalescingInfo coal;
        coal.instructionAddr = inst.address;
        // Heuristic: if the assembly mentions threadIdx-derived register,
        // it's likely coalesced
        if (inst.assembly.find("R0") != std::string::npos ||
            inst.assembly.find("%threadIdx") != std::string::npos) {
          coal.accessPattern = "coalesced";
          coal.transactionsNeeded = 1;
          coal.description = "Sequential thread access (1 transaction)";
        } else {
          coal.accessPattern = "likely_coalesced";
          coal.transactionsNeeded = 1;
          coal.description = "Register-based access";
        }
        if (opcode == 0b1010 || opcode == 0b1011) {
          coal.description += " [shared memory - low latency]";
        }
        analysis.coalescing.push_back(coal);
      }
      break;
    case 0b1100: // BAR
      analysis.barrierCount++;
      break;
    default:
      if (opcode != 0b1111 && opcode != 0b1001 && opcode != 0b0000)
        analysis.computeInstructions++;
      break;
    }
  }
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

  trace.irStages.push_back(
      {"Frontend \xe2\x86\x92 TinyGPU Dialect", captureIR(*module)});

  if (opts.format == OutputFormat::MLIR) {
    module->print(os);
    return trace;
  }

  // Stage 2.5: Optimization Passes
  OptimizationStats optStats;
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      optStats = runAllOptimizations(&op);
    }
  }

  trace.irStages.push_back(
      {"Optimization Passes", captureIR(*module)});
  trace.analysis.optimizationSummary = optStats.summary();

  // Stage 3: Register Allocation
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      if (failed(allocateRegisters(&op))) {
        llvm::errs() << "Register allocation failed\n";
        return trace;
      }
    }
  }

  // Count registers used
  int maxReg = 0;
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      op.walk([&](Operation *child) {
        if (auto rdAttr = child->getAttrOfType<IntegerAttr>("rd")) {
          int reg = rdAttr.getInt();
          if (reg < 13 && reg > maxReg)
            maxReg = reg;
        }
      });
    }
  }
  trace.analysis.registersUsed = maxReg + 1;

  trace.irStages.push_back({"Register Allocation", captureIR(*module)});

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

  // Run analysis on compiled output
  analyzeInstructions(trace);

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
