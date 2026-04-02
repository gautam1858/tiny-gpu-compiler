add_mlir_library(TGCPasses
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiny-gpu-compiler/Passes

  LINK_LIBS PUBLIC
  MLIRIR
  TinyGPUDialect
)
