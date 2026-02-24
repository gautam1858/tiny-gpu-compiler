#ifndef TINYGPU_DIALECT_TD
#define TINYGPU_DIALECT_TD

include "mlir/IR/OpBase.td"

def TinyGPU_Dialect : Dialect {
  let name = "tinygpu";
  let summary = "Dialect targeting the tiny-gpu educational processor";
  let description = [{
    The TinyGPU dialect provides operations that map directly to the tiny-gpu
    hardware's 16-bit ISA. This dialect serves as the target for lowering from
    MLIR's standard GPU dialect, enabling an educational compiler pipeline from
    high-level GPU kernels to binary instructions.

    The tiny-gpu processor features:
    - 11 instructions (16-bit encoding)
    - 13 general-purpose 8-bit registers (R0-R12)
    - 3 read-only special registers (R13=%blockIdx, R14=%blockDim, R15=%threadIdx)
    - 256-byte data memory, 256-entry program memory
    - SIMD-like execution: all threads in a block run in lockstep
  }];
  let cppNamespace = "::mlir::tinygpu";
  let useDefaultTypePrinterParser = 1;
}

#endif // TINYGPU_DIALECT_TD
