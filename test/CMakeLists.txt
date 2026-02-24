configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TGC_TEST_DEPENDS
  tgc
)

add_lit_testsuite(check-tgc "Running tiny-gpu-compiler regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TGC_TEST_DEPENDS}
)
set_target_properties(check-tgc PROPERTIES FOLDER "Tests")
