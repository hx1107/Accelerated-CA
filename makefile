CC=g++
NVCC=nvcc
CUDAFLAGS= -O3 -std=c++11 -ccbin gcc-7
LIBS= -lm -lstdc++

example: example.cu
	$(NVCC) example.cu -o example $(CUDAFLAGS) $(LIBS)

test: example
	./example

clean:
	rm -rf *.o example
