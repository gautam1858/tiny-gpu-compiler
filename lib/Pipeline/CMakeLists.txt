add_mlir_library(TGCPipeline
  Pipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiny-gpu-compiler/Pipeline

  LINK_LIBS PUBLIC
  MLIRIR
  TinyGPUDialect
  TGCFrontend
  TGCCodeGen
)
