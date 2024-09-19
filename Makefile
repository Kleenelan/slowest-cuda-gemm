
#src/dgemm.cu

CU_SRC := src/dgemm.cu
CPP_SRC :=  src/hello_dgemm.cpp src/init_matrix.cpp src/dgemm.cpp

CU_OBJ := $(CU_SRC:.cu=.cu.o)
CPP_OBJ := $(CPP_SRC:.cpp=.cpp.o)

hello: $(CPP_OBJ) $(CU_OBJ)
	nvcc -g -Xcompiler -fopenmp  $^ -o $@ $(INC) $(LD_FLAGS) -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

%.cpp.o: %.cpp
	nvcc -g -Xcompiler -fopenmp -c $< -o $@ $(INC)

%.cu.o: %.cu
	nvcc -g -dlink   -c $< -o $@ $(INC) -L/usr/local/cuda/lib64 -lcudart -lcudadevrt


INC := -I inc/ -I /usr/local/cuda/include
LD_FLAGS := -L /usr/local/cuda/lib64 -L ../thirdparts/openblas/local/lib -lcudart -lopenblas -lgfortran



.PHONY: clean
clean:
	-rm -rf $(CU_OBJ) $(CPP_OBJ) hello


