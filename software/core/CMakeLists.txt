cmake_minimum_required(VERSION 3.15)
project(libuap CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libuap is header-only; expose it as an INTERFACE library.
add_library(libuap INTERFACE)
target_include_directories(libuap INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Native example / smoke test.
add_executable(uap_example example_main.cpp)
target_link_libraries(uap_example PRIVATE libuap)

if(MSVC)
  target_compile_options(uap_example PRIVATE /O2)
else()
  target_compile_options(uap_example PRIVATE -O2)
endif()
