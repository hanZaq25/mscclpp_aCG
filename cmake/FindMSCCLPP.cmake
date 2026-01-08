# FindMSCCLPP.cmake

find_path(MSCCLPP_INCLUDE_DIR
  NAMES mscclpp/nccl.h
  HINTS ${MSCCLPP_HOME}/include $ENV{MSCCLPP_HOME}/include $ENV{HOME}/.local/include)

find_library(MSCCLPP_LIBRARY
  NAMES mscclpp_nccl
  HINTS ${MSCCLPP_HOME}/lib $ENV{MSCCLPP_HOME}/lib $ENV{HOME}/.local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MSCCLPP DEFAULT_MSG
  MSCCLPP_INCLUDE_DIR MSCCLPP_LIBRARY)

mark_as_advanced(MSCCLPP_INCLUDE_DIR MSCCLPP_LIBRARY)