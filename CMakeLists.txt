cmake_minimum_required(VERSION 3.20)
project(tiny-gpu-compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLIR (requires MLIR_DIR or LLVM_DIR to be set)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Optional CIRCT support for Backend 2 (Verilog generation)
option(TGC_ENABLE_CIRCT "Enable CIRCT backend for Verilog generation" OFF)
if(TGC_ENABLE_CIRCT)
  find_package(CIRCT REQUIRED CONFIG)
  include_directories(${CIRCT_INCLUDE_DIRS})
  add_definitions(-DTGC_ENABLE_CIRCT)
  message(STATUS "CIRCT backend enabled")
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# Testing
option(TGC_ENABLE_TESTS "Enable tests" ON)
if(TGC_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
