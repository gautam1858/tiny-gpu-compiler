add_mlir_dialect_library(TinyGPUDialect
  TinyGPUDialect.cpp
  TinyGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiny-gpu-compiler/Dialect/TinyGPU

  DEPENDS
  TinyGPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
