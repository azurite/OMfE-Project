################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables
CU_SRCS += \
../src/CudaMatrixMultiply.cu \
../src/CudaMatrixMultiplyShared.cu \
../src/main.cu

OBJS += \
./src/CudaMatrixMultiply.o \
./src/CudaMatrixMultiplyShared.o \
./src/main.o

CU_DEPS += \
./src/CudaMatrixMultiply.d \
./src/CudaMatrixMultiplyShared.d \
./src/main.d

NVCC_COMPILER = /usr/local/cuda-10.2/bin/nvcc

ifdef archlinux
NVCC_COMPILER = /opt/cuda/bin/nvcc
LIBS := -I ../../../lib/
endif

# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	$(NVCC_COMPILER) -lineinfo -O3 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<" $(LIBS)
	$(NVCC_COMPILER) -lineinfo -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<" --compiler-options=" -O3  -march=native -pipe -fopenmp" $(LIBS)
	@echo 'Finished building: $<'
	@echo ' '
