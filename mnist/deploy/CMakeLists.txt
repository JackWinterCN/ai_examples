cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
 
project(minist_test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
# find_package(cuDNN REQUIRED)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50)

# 编译CUDA源文件
cuda_add_library(cuda_core fp16_dev.cu)
 
# 编译C++源文件
add_executable(keras_minist myMnistCUDNN.cpp fp16_emu.cpp)
include_directories(/usr/local/cuda/include /usr/include)
# 链接CUDA目标到C++可执行文件
target_link_libraries(keras_minist PRIVATE cuda_core 
/usr/local/cuda/lib64/libcublasLt.so
/usr/local/cuda/lib64/libcudart.so
/usr/local/cuda/lib64/libcublas.so
/usr/lib/x86_64-linux-gnu/libcudnn.so
/usr/lib/x86_64-linux-gnu/libfreeimage.so)

# 编译C++源文件
add_executable(caffe_minist mnistCUDNN.cpp fp16_emu.cpp)
# 链接CUDA目标到C++可执行文件
target_link_libraries(caffe_minist PRIVATE cuda_core 
/usr/local/cuda/lib64/libcublasLt.so
/usr/local/cuda/lib64/libcudart.so
/usr/local/cuda/lib64/libcublas.so
/usr/lib/x86_64-linux-gnu/libcudnn.so
/usr/lib/x86_64-linux-gnu/libfreeimage.so)


# enable_language(CUDA)
# set(CUDA_GEN_CODE "-gencode=arch=compute_50,code=sm_50")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
# set_source_files_properties(mnistCUDNN.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
# cuda_add_executable(mnistTest ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.cu)
