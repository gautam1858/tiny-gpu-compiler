get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(tgc
  tgc.cpp
)

target_link_libraries(tgc PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
  TinyGPUDialect
  TGCFrontend
  TGCCodeGen
  TGCPipeline
)

llvm_update_compile_flags(tgc)
mlir_check_all_link_libraries(tgc)
