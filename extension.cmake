# ricursive (Ripple-65B family). Host side: config, weight table, tokenizer,
# chat template, executor contract; scalar CPU reference and CPU executor for
# validation; vendored projection kernels in their own HIP target.
gufo_extension_target(
  NAME gufo_ricursive
  SOURCES
    src/config.cpp
    src/executor_iface.cpp
    src/extension.cpp
    src/probe.cpp
    src/text/chat_template.cpp
    src/text/json_value.cpp
    src/text/tokenizer.cpp
    src/weights.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
    ICU::uc
)

# Projections and embedding (docs/S2-PROJECT.md). The hip/ggml units are
# excerpts of Gufo kernels and keep the Gemma 4 extension's flags for the
# same files: the configuration's HIP flags without fast math for gemv*,
# -fno-slp-vectorize for the blocked K-quant prefill tiles, wave64 for the
# wave64 tiles. Nothing here may get -ffast-math (decode GEMV and the
# small-batch kernels must round identically).
gufo_extension_target(
  NAME gufo_ricursive_hip
  SOURCES
    hip/embed.cpp
    hip/project.cpp
  LINK_LIBRARIES
    gufo_ricursive
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
)
if(TARGET gufo_ricursive_hip)
  gufo_extension_hip_sources(gufo_ricursive_hip
    SOURCES
      hip/ggml/gemv.hip.cpp
      hip/ggml/gemv_quant.hip.cpp
  )
  gufo_extension_hip_sources(gufo_ricursive_hip
    SOURCES
      hip/ggml/prefill_quant_gemm.hip.cpp
    OPTIONS -fno-slp-vectorize
  )
  gufo_extension_hip_sources(gufo_ricursive_hip
    SOURCES
      hip/ggml/prefill_quant_wave64.hip.cpp
    OPTIONS -mwavefrontsize64 -fno-slp-vectorize
  )
  target_compile_options(gufo_ricursive_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()

# The scalar oracle decodes every weight on the fly; unoptimized it is
# unusable even as an oracle, so it keeps -O2 in every configuration. Only
# validation targets build it.
gufo_extension_target(
  NAME gufo_ricursive_reference
  SOURCES
    src/cpu_executor.cpp
    src/reference.cpp
  LINK_LIBRARIES
    gufo_ricursive
    gufo_ext_runtime
    gufo_ext_core
)
if(TARGET gufo_ricursive_reference)
  target_compile_options(gufo_ricursive_reference PRIVATE -O2)
  # Without OpenMP its parallel loops silently run on one thread.
  find_package(OpenMP REQUIRED)
  target_link_libraries(gufo_ricursive_reference PRIVATE OpenMP::OpenMP_CXX)
endif()

# Tests (CPU only). `ricursive_tests` builds every test binary so ctest never
# runs a stale one. Fixture locations default to the staging tree and can be
# overridden with RICURSIVE_FIXTURES / RICURSIVE_ORACLE_OUT / RICURSIVE_S1_DIR.
if(BUILD_TESTING AND NOT TARGET ricursive_tests)
  add_custom_target(ricursive_tests)
endif()

set(RICURSIVE_STAGE "/server/staging/ripple-gufo" CACHE PATH
  "ricursive fixtures and oracle files (fixtures/, oracle/, ref/hf/)")

# _ricursive_test(name [LABELS l...] [ARGS a...] [TIMEOUT s] SOURCES ...
#                 LINK_LIBRARIES ...): one binary, one ctest entry. Labels
# default to "cpu"; GPU tests are labelled "gpu" and skip with 77.
function(_ricursive_test name)
  cmake_parse_arguments(ARG "NO_TEST" "TIMEOUT"
    "SOURCES;LINK_LIBRARIES;LABELS;ARGS" ${ARGN})
  set(target ricursive_${name}_test)
  gufo_extension_target(
    NAME ${target}
    SOURCES ${ARG_SOURCES}
    LINK_LIBRARIES ${ARG_LINK_LIBRARIES} gufo_ext_runtime gufo_ext_core
  )
  if(NOT TARGET ${target})
    return()
  endif()
  target_compile_options(${target} PRIVATE -O2)
  add_dependencies(ricursive_tests ${target})
  if(ARG_NO_TEST)
    return()
  endif()
  if(NOT ARG_LABELS)
    set(ARG_LABELS cpu)
  endif()
  add_test(NAME ricursive.${name} COMMAND ${target} ${ARG_ARGS})
  set_tests_properties(ricursive.${name} PROPERTIES
    LABELS "ricursive;${ARG_LABELS}" SKIP_RETURN_CODE 77
    ENVIRONMENT "OMP_NUM_THREADS=4")
  if(ARG_TIMEOUT)
    set_tests_properties(ricursive.${name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
endfunction()

_ricursive_test(config
  SOURCES tests/config_test.cpp LINK_LIBRARIES gufo_ricursive)
_ricursive_test(weights
  SOURCES tests/weights_test.cpp LINK_LIBRARIES gufo_ricursive)
_ricursive_test(oracle TIMEOUT 1800
  SOURCES tests/oracle_test.cpp
  LINK_LIBRARIES gufo_ricursive_reference gufo_ricursive)
_ricursive_test(quant TIMEOUT 1800
  SOURCES tests/quant_test.cpp
  LINK_LIBRARIES gufo_ricursive_reference gufo_ricursive)
_ricursive_test(incremental TIMEOUT 1800
  SOURCES tests/incremental_test.cpp
  LINK_LIBRARIES gufo_ricursive_reference gufo_ricursive)

_ricursive_test(executor_contract TIMEOUT 1800
  SOURCES tests/executor_contract_test.cpp tests/executor_backends_cpu.cpp
  LINK_LIBRARIES gufo_ricursive_reference gufo_ricursive)

# Tokenizer and chat-template goldens (src/text/README.md). The seeded
# property runs and the all-code-point sweeps need Python oracles and stay in
# tests/text/Makefile.
_ricursive_test(tokenizer
  SOURCES tests/text/test_tokenizer.cpp LINK_LIBRARIES gufo_ricursive
  ARGS goldens "${RICURSIVE_STAGE}/fixtures/vocab-only.gguf"
       "${RICURSIVE_STAGE}/oracle/tok-goldens.jsonl")
_ricursive_test(chat
  SOURCES tests/text/test_chat.cpp LINK_LIBRARIES gufo_ricursive
  ARGS cases "${RICURSIVE_STAGE}/oracle/chat-goldens.jsonl")
if(TARGET ricursive_chat_test)
  # The GGUF's template text against ref/hf/chat_template.jinja.
  add_test(NAME ricursive.chat_template_text COMMAND ricursive_chat_test
    template "${RICURSIVE_STAGE}/fixtures/vocab-only.gguf"
    "${RICURSIVE_STAGE}/ref/hf/chat_template.jinja")
  set_tests_properties(ricursive.chat_template_text PROPERTIES
    LABELS "ricursive;cpu" SKIP_RETURN_CODE 77)
endif()

# Embedding: the default run is host-only (no HIP call); `gpu` also copies
# rows to the device. Projections: one ctest per case, GPU only. Run gpu
# tests only through the GPU admission lock (docs/S2-PROJECT.md).
_ricursive_test(embed
  SOURCES tests/embed_test.cpp
  LINK_LIBRARIES gufo_ricursive_hip gufo_ricursive gufo_ext_hip)
if(TARGET ricursive_embed_test)
  target_compile_options(ricursive_embed_test PRIVATE
    -ffp-contract=off -fno-fast-math)
  add_test(NAME ricursive.embed_gpu COMMAND ricursive_embed_test gpu)
  set_tests_properties(ricursive.embed_gpu PROPERTIES
    LABELS "ricursive;gpu" SKIP_RETURN_CODE 77)
endif()
set(_ricursive_projection_cases q4k_10240x5120 q8_0_1024x5120
  q6k_5120x10240 q4k_40960x5120 q5k_40960x5120 q4k_5120x40960
  q5k_5120x40960 q6k_100352x5120 f32-tiny twins reject)
_ricursive_test(projection_ops NO_TEST
  SOURCES tests/projection_ops_test.cpp
  LINK_LIBRARIES gufo_ricursive_hip gufo_ricursive gufo_ext_hip)
if(TARGET ricursive_projection_ops_test)
  # Exact reference arithmetic in the test itself (as built standalone).
  target_compile_options(ricursive_projection_ops_test PRIVATE
    -ffp-contract=off -fno-fast-math)
  foreach(case IN LISTS _ricursive_projection_cases)
    add_test(NAME ricursive.projection_ops.${case}
      COMMAND ricursive_projection_ops_test ${case})
    set_tests_properties(ricursive.projection_ops.${case} PROPERTIES
      LABELS "ricursive;gpu" SKIP_RETURN_CODE 77 RESOURCE_LOCK ricursive_gpu)
  endforeach()
endif()

# Tool, built on request: ricursive_ref_logits MODEL.gguf TOKENS.i32 OUT.f32
gufo_extension_target(
  NAME ricursive_ref_logits
  SOURCES tools/ref_logits.cpp
  LINK_LIBRARIES gufo_ricursive_reference gufo_ricursive gufo_ext_runtime
    gufo_ext_core
)
if(TARGET ricursive_ref_logits)
  set_target_properties(ricursive_ref_logits PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/ricursive)

# --- WP-D engine begin ---
# Engine (Model/Session), component-cache adapter, runner, provider, drivers,
# output parser (docs/S2-ENGINE.md). The sources join gufo_ricursive; the CPU
# executor (gufo_ricursive_reference) is linked in so `gufo prompt` can run a
# tiny model with RICURSIVE_BACKEND=cpu. It is for tests and tiny models only.
option(RICURSIVE_CPU_BACKEND
  "Build the scalar CPU executor into gufo_ricursive (tests and tiny models)"
  ${BUILD_TESTING})
if(TARGET gufo_ricursive)
  target_sources(gufo_ricursive PRIVATE
    ${GUFO_EXTENSION_BUILD_ROOT}/src/chat_prompt.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/chat_request.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/continuation_adapter.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/drivers.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/engine.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/executor_factory.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/output_parser.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/provider.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/src/runner.cpp)
  target_link_libraries(gufo_ricursive PUBLIC gufo_ext_cache)
  if(TARGET gufo_ext_hip)
    # The component cache's pinned-host backing is HIP only.
    target_compile_definitions(gufo_ricursive PRIVATE RICURSIVE_HIP_CACHE=1)
    target_link_libraries(gufo_ricursive PUBLIC gufo_ext_hip)
  endif()
  if(RICURSIVE_CPU_BACKEND AND TARGET gufo_ricursive_reference)
    target_compile_definitions(gufo_ricursive PRIVATE RICURSIVE_WITH_CPU_BACKEND=1)
    target_link_libraries(gufo_ricursive PUBLIC gufo_ricursive_reference)
  endif()
  # The GPU executor (separate work package) defines
  # RICURSIVE_HAVE_GPU_EXECUTOR and provides CreateGpuExecutor, see
  # src/executor_factory.hpp.
endif()

_ricursive_test(output_parser
  SOURCES tests/output_parser_test.cpp LINK_LIBRARIES gufo_ricursive)
_ricursive_test(engine TIMEOUT 1800
  SOURCES tests/engine_test.cpp
  LINK_LIBRARIES gufo_ricursive gufo_ricursive_reference gufo_ext_cache)
_ricursive_test(continuation TIMEOUT 1800
  SOURCES tests/continuation_test.cpp
  LINK_LIBRARIES gufo_ricursive gufo_ricursive_reference gufo_ext_cache)
# The chat fixture (real tokenizer and template, tiny random weights) is
# generated into the build directory by a committed script, as a ctest fixture.
set(RICURSIVE_CHAT_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/chat-tiny.gguf")
_ricursive_test(chat_e2e TIMEOUT 1800
  SOURCES tests/chat_e2e_test.cpp
  LINK_LIBRARIES gufo_ricursive gufo_ricursive_reference gufo_ext_cache
  ARGS "${RICURSIVE_CHAT_FIXTURE}" "${RICURSIVE_STAGE}/oracle/chat-goldens.jsonl"
       "${RICURSIVE_STAGE}/fixtures/vocab-only.gguf")
if(TARGET ricursive_chat_e2e_test)
  add_test(NAME ricursive.chat_fixture
    COMMAND python3 "${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures_text/make_chat_fixture.py"
            "${RICURSIVE_STAGE}/fixtures/vocab-only.gguf" "${RICURSIVE_CHAT_FIXTURE}")
  set_tests_properties(ricursive.chat_fixture PROPERTIES
    FIXTURES_SETUP ricursive_chat_fixture LABELS "ricursive;cpu")
  set_tests_properties(ricursive.chat_e2e PROPERTIES
    FIXTURES_REQUIRED ricursive_chat_fixture)
endif()
# --- WP-D engine end ---
