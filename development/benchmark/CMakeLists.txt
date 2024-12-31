cmake_minimum_required(VERSION 3.18)
project(benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wno-deprecated-declarations")

find_package(CUDA REQUIRED)
find_library(TENSORRT_LIB nvinfer)
find_library(PARSER_LIB nvonnxparser)
find_package(Eigen3 3.1.0 REQUIRED)

find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

set(TENSORRTBUFFER_SOURCE_DIR "../../Thirdparty/tensorrtbuffer")
link_directories(${TENSORRTBUFFER_SOURCE_DIR}/lib)

include_directories(
    ${CUDA_INCLUDE_DIRS} 
    ${EIGEN3_INCLUDE_DIR}
    ${TENSORRTBUFFER_SOURCE_DIR}/include
    /usr/include/x86_64-linux-gnu
)

add_executable(
    ${PROJECT_NAME} 
    main.cpp
    SuperPoint.cpp
    LightGlue.cpp
)

target_link_libraries(
    ${PROJECT_NAME} 
    ${CUDA_LIBRARIES} 
    ${TENSORRT_LIB} 
    ${PARSER_LIB}
    ${EIGEN3_LIBS}
    ${OpenCV_LIBS}
    tensorrtbuffer
)
