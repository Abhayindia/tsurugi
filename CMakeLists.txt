cmake_minimum_required(VERSION 3.16)
project(tsurugi CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TSURUGI_NATIVE "Tune for the build machine (-march=native)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(TSURUGI_NATIVE)
    add_compile_options(-march=native)
  else()
    add_compile_options(-mpopcnt)
  endif()
endif()

add_library(tsurugi_core STATIC
  src/bitboard.cpp
  src/position.cpp
  src/movegen.cpp
  src/perft.cpp
)
target_include_directories(tsurugi_core PUBLIC src)

enable_testing()

add_executable(perft_test tests/perft_test.cpp)
target_link_libraries(perft_test PRIVATE tsurugi_core)
add_test(NAME perft_quick COMMAND perft_test --quick)

add_executable(divide tests/divide.cpp)
target_link_libraries(divide PRIVATE tsurugi_core)
