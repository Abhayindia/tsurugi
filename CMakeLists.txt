cmake_minimum_required(VERSION 3.16)
project(tsurugi CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TSURUGI_NATIVE "Tune for the build machine (-march=native)" OFF)
set(TSURUGI_NNUE_HIDDEN 1536 CACHE STRING "Hidden layer size of the network")
add_compile_definitions(TSURUGI_NNUE_HIDDEN=${TSURUGI_NNUE_HIDDEN})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(TSURUGI_NATIVE)
    add_compile_options(-march=native)
    option(TSURUGI_USE_PEXT "Use BMI2 PEXT for slider attacks" ON)
    if(TSURUGI_USE_PEXT)
      add_compile_definitions(TSURUGI_PEXT)
    endif()
  else()
    add_compile_options(-mpopcnt)
  endif()
endif()

add_library(tsurugi_core STATIC
  src/bitboard.cpp
  src/position.cpp
  src/movegen.cpp
  src/perft.cpp
  src/eval.cpp
  src/nnue.cpp
  src/tt.cpp
  src/search.cpp
  src/datagen.cpp
  src/uci.cpp
)
target_include_directories(tsurugi_core PUBLIC src)

find_package(Threads REQUIRED)
target_link_libraries(tsurugi_core PUBLIC Threads::Threads)

add_executable(tsurugi src/main.cpp)
target_link_libraries(tsurugi PRIVATE tsurugi_core)

# The engine loads tsurugi.nnue from its own directory; ship the default network with every build.
set(TSURUGI_DEFAULT_NET "${CMAKE_SOURCE_DIR}/nets/tsurugi-v7.nnue" CACHE FILEPATH "Network copied next to the binary")
if(EXISTS "${TSURUGI_DEFAULT_NET}")
  configure_file("${TSURUGI_DEFAULT_NET}" "${CMAKE_BINARY_DIR}/tsurugi.nnue" COPYONLY)
endif()

enable_testing()

add_executable(perft_test tests/perft_test.cpp)
target_link_libraries(perft_test PRIVATE tsurugi_core)
add_test(NAME perft_quick COMMAND perft_test --quick)

add_executable(eval_test tests/eval_test.cpp)
target_link_libraries(eval_test PRIVATE tsurugi_core)
add_test(NAME eval_symmetry COMMAND eval_test)

add_executable(divide tests/divide.cpp)
target_link_libraries(divide PRIVATE tsurugi_core)
