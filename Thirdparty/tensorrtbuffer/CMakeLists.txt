cmake_minimum_required(VERSION 3.6)
project(tensorrtbuffer)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

message(STATUS "NV_IS_SAFETY: ${NV_IS_SAFETY}")
message(status, ${PROJECT_SOURCE_DIR})
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(
    tensorrtbuffer 
    SHARED 
    src/logger.cpp 
)

target_link_libraries(tensorrtbuffer
 nvinfer
 nvinfer_plugin
 nvonnxparser
 ${OpenCV_LIBRARIES}
 ${CUDA_LIBRARIES}
${PROJECT_SOURCE_DIR}/include

)

