#include "tiny-gpu-compiler/Pipeline/Pipeline.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                           cl::desc("<input .tgc file>"),
                                           cl::Required);

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                            cl::value_desc("filename"),
                                            cl::init("-"));

static cl::opt<tgc::OutputFormat> emitFormat(
    "emit", cl::desc("Output format"),
    cl::values(
        clEnumValN(tgc::OutputFormat::MLIR, "mlir", "MLIR IR"),
        clEnumValN(tgc::OutputFormat::Assembly, "asm", "Annotated assembly"),
        clEnumValN(tgc::OutputFormat::Hex, "hex",
                   "Hex text (one instruction per line)"),
        clEnumValN(tgc::OutputFormat::Binary, "bin", "Raw binary"),
        clEnumValN(tgc::OutputFormat::JsonTrace, "trace",
                   "JSON trace for web visualizer")),
    cl::init(tgc::OutputFormat::Assembly));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                               "tiny-gpu-compiler: compile .tgc kernels to "
                               "tiny-gpu binary instructions\n");

  // Read input file
  auto fileOrErr = MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    errs() << "Error reading " << inputFilename << ": " << ec.message() << "\n";
    return 1;
  }

  std::string source = (*fileOrErr)->getBuffer().str();

  // Set up output
  std::error_code ec;
  auto output = std::make_unique<ToolOutputFile>(outputFilename, ec,
                                                  sys::fs::OF_None);
  if (ec) {
    errs() << "Error opening output: " << ec.message() << "\n";
    return 1;
  }

  // Compile
  tgc::CompilerOptions opts;
  opts.format = emitFormat;
  tgc::compile(source, opts, output->os());

  output->keep();
  return 0;
}
