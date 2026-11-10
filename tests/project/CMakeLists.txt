# Standalone build of ricursive work package A (projections + embedding) against the prebuilt public
# gufo libraries. Not part of extension.cmake. Configure and build:
#   source /server/programming/gufo-ripple/env.sh
#   cmake -S tests/project -B /server/staging/ripple-gufo/s2-project/build -G Ninja
#   nice -n 10 cmake --build /server/staging/ripple-gufo/s2-project/build -j2
# Run the GPU tests only through run-gpu-tests.sh (GPU admission lock).
cmake_minimum_required(VERSION 3.24)

set(GUFO_SRC "/server/programming/gufo-ripple/gufo" CACHE PATH "gufo source tree (public headers)")
set(GUFO_BUILD "/server/programming/gufo-ripple/build-release" CACHE PATH "built gufo (read-only)")
set(ROCM "/opt/rocm-7.2.2" CACHE PATH "ROCm install")
set(CMAKE_HIP_COMPILER "${ROCM}/lib/llvm/bin/clang++" CACHE FILEPATH "")
set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(ricursive_project LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
get_filename_component(EXT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

list(APPEND CMAKE_PREFIX_PATH "${ROCM}")
find_package(hip REQUIRED CONFIG)
find_package(hipblas REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# What gufo's own HIP targets compile with (build-release/build.ninja, gufo_qwen_kernels):
#   -O3 -DNDEBUG -std=c++20 --offload-arch=gfx1151 -Wall -Wextra -Wpedantic
#   -DENGINE_ENABLE_HIP=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
set(RIC_DEFS ENGINE_ENABLE_HIP=1 USE_PROF_API=1 __HIP_PLATFORM_AMD__=1 __HIP_ROCclr__=1)
set(RIC_INCLUDES "${EXT_ROOT}" "${GUFO_SRC}" "${GUFO_SRC}/include")
set(RIC_SYS_INCLUDES "${ROCM}/include")

# The vendored kernels and the two host/device sources, in their own target. Per-source flags are the
# Gemma 4 extension's for these files: the configuration's default HIP flags (no fast math) for gemv*,
# -fno-slp-vectorize for the blocked K-quant prefill tiles, -mwavefrontsize64 -fno-slp-vectorize for the
# wave64 tiles.
add_library(ricursive_project STATIC
  ${EXT_ROOT}/hip/ggml/gemv.hip.cpp
  ${EXT_ROOT}/hip/ggml/gemv_quant.hip.cpp
  ${EXT_ROOT}/hip/ggml/prefill_quant_gemm.hip.cpp
  ${EXT_ROOT}/hip/ggml/prefill_quant_wave64.hip.cpp
  ${EXT_ROOT}/hip/project.cpp
  ${EXT_ROOT}/hip/embed.cpp)
set_source_files_properties(
  ${EXT_ROOT}/hip/ggml/gemv.hip.cpp ${EXT_ROOT}/hip/ggml/gemv_quant.hip.cpp
  ${EXT_ROOT}/hip/ggml/prefill_quant_gemm.hip.cpp ${EXT_ROOT}/hip/ggml/prefill_quant_wave64.hip.cpp
  ${EXT_ROOT}/hip/project.cpp ${EXT_ROOT}/hip/embed.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(${EXT_ROOT}/hip/ggml/prefill_quant_gemm.hip.cpp
  PROPERTIES COMPILE_OPTIONS "-fno-slp-vectorize")
set_source_files_properties(${EXT_ROOT}/hip/ggml/prefill_quant_wave64.hip.cpp
  PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64;-fno-slp-vectorize")
target_compile_definitions(ricursive_project PUBLIC ${RIC_DEFS})
target_include_directories(ricursive_project PUBLIC ${RIC_INCLUDES})
target_include_directories(ricursive_project SYSTEM PUBLIC ${RIC_SYS_INCLUDES})
target_compile_options(ricursive_project PRIVATE -O3 -DNDEBUG --offload-arch=gfx1151
  -Wall -Wextra -Wpedantic)

# Public gufo libraries (static, read-only from the prebuilt tree). libgufo_core.a holds
# gufo::quant::DequantizeRow and the GGUF reader; it needs the platform library, the Rust staticlib
# (src/core/rust), the HIP host helpers and OpenSSL. The archives reference each other, so the group is
# rescanned.
set(GUFO_LIBS
  ${GUFO_BUILD}/libgufo_core.a
  ${GUFO_BUILD}/libgufo_hip_host.a
  ${GUFO_BUILD}/src/core/platform/libgufo_platform.a
  ${GUFO_BUILD}/src/core/rust/cargo/release/libgufo_rust.a)
foreach(lib IN LISTS GUFO_LIBS)
  if(NOT EXISTS "${lib}")
    message(FATAL_ERROR "missing prebuilt gufo library: ${lib}")
  endif()
endforeach()
add_library(gufo_public INTERFACE)
target_link_libraries(gufo_public INTERFACE
  "$<LINK_GROUP:RESCAN,${GUFO_LIBS}>"
  hip::host roc::hipblas OpenSSL::Crypto Threads::Threads ${CMAKE_DL_LIBS})

target_link_libraries(ricursive_project PUBLIC gufo_public)

function(ric_test name source)
  add_executable(${name} ${source})
  set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
  target_link_libraries(${name} PRIVATE ricursive_project)
  # Exact reference arithmetic: no FMA contraction, no fast math.
  target_compile_options(${name} PRIVATE -O2 -ffp-contract=off -fno-fast-math --offload-arch=gfx1151
    -Wall -Wextra)
endfunction()

ric_test(ricursive_embed_test ${EXT_ROOT}/tests/embed_test.cpp)
ric_test(ricursive_projection_ops_test ${EXT_ROOT}/tests/projection_ops_test.cpp)
