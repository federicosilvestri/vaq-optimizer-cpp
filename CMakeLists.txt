cmake_minimum_required(VERSION 3.13)
project(optimizer)

set(CMAKE_CXX_STANDARD 11)
add_executable(optimizer
        src/main.cpp
        src/json.hpp
)
target_link_libraries(optimizer ${GLPK_LIBRARIES})
target_link_libraries(optimizer glpk)
