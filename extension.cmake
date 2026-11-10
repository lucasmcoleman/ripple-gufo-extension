# ricursive: empty skeleton. Probe only; no runner, no GPU kernels yet.
gufo_extension_target(
  NAME gufo_ricursive
  SOURCES
    src/extension.cpp
    src/probe.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_ext_core
)

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/ricursive)
