set(LLVM_TARGET_DEFINITIONS TinyGPUOps.td)

mlir_tablegen(TinyGPUOps.h.inc -gen-op-decls)
mlir_tablegen(TinyGPUOps.cpp.inc -gen-op-defs)
mlir_tablegen(TinyGPUDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TinyGPUDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(TinyGPUOpsIncGen)
