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


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -lineinfo -O3 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -lineinfo -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<" --compiler-options=" -O3  -march=native -pipe -fopenmp"
	@echo 'Finished building: $<'
	@echo ' '


