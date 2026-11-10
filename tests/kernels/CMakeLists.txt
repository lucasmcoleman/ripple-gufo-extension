# Standalone build of the ricursive GPU kernels and their tests, without gufo.
#
#   source /server/programming/gufo-ripple/env.sh
#   cmake -S tests/kernels -B build-kernels -G Ninja \
#         -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.2/lib/llvm/bin/clang++
#   nice -n 10 cmake --build build-kernels -j2
#   tests/kernels/run-gpu-tests.sh build-kernels
#
# The math test is host-only (plain C++, no HIP). The other tests need a GPU
# and must run through the admission lock (run-gpu-tests.sh does that).
cmake_minimum_required(VERSION 3.21)
project(ricursive_kernels LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()

get_filename_component(RIC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
find_package(hip REQUIRED CONFIG PATHS /opt/rocm-7.2.2 /opt/rocm-7.2.2/lib/cmake/hip)

# Same flags as the Gemma 4 extension's strict-FP kernel units (-O3, no fast
# math, no contraction): a row's result must not depend on the rest of its
# launch.
set(RIC_HIP_FLAGS -O3 -fno-fast-math -ffp-contract=off -Wno-unused-command-line-argument)

add_library(ricursive_kernels STATIC ${RIC_ROOT}/hip/kernels.hip.cpp)
set_source_files_properties(${RIC_ROOT}/hip/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(ricursive_kernels PRIVATE ${RIC_HIP_FLAGS})
target_include_directories(ricursive_kernels PUBLIC ${RIC_ROOT})
target_link_libraries(ricursive_kernels PUBLIC hip::host)

add_executable(ricursive_kernels_math_test ${RIC_ROOT}/tests/kernels_math_test.cpp)
target_include_directories(ricursive_kernels_math_test PRIVATE ${RIC_ROOT})
target_compile_options(ricursive_kernels_math_test PRIVATE -O2)

# GPU tests: host code only (the kernels live in the library), built as HIP so
# hip_runtime.h resolves; -O2.
foreach(name kernel_ops attention_ops)
  add_executable(ricursive_${name}_test ${RIC_ROOT}/tests/${name}_test.cpp)
  set_source_files_properties(${RIC_ROOT}/tests/${name}_test.cpp PROPERTIES LANGUAGE HIP)
  target_compile_options(ricursive_${name}_test PRIVATE -O2)
  target_link_libraries(ricursive_${name}_test PRIVATE ricursive_kernels)
endforeach()
