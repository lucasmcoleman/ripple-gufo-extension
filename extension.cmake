# ricursive (Ripple-65B family). Host side: config, weight table, scalar CPU
# reference. No GPU code yet.
gufo_extension_target(
  NAME gufo_ricursive
  SOURCES
    src/config.cpp
    src/extension.cpp
    src/probe.cpp
    src/weights.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)

# The scalar oracle decodes every weight on the fly; unoptimized it is
# unusable even as an oracle, so it keeps -O2 in every configuration. Only
# validation targets build it.
gufo_extension_target(
  NAME gufo_ricursive_reference
  SOURCES
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

function(_ricursive_test name)
  cmake_parse_arguments(ARG "" "TIMEOUT" "SOURCES;LINK_LIBRARIES" ${ARGN})
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
  add_test(NAME ricursive.${name} COMMAND ${target})
  set_tests_properties(ricursive.${name} PROPERTIES
    LABELS "ricursive;cpu" SKIP_RETURN_CODE 77
    ENVIRONMENT "OMP_NUM_THREADS=4")
  if(ARG_TIMEOUT)
    set_tests_properties(ricursive.${name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
  add_dependencies(ricursive_tests ${target})
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
