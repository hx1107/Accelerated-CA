CC=g++
NVCC=nvcc
CUDAFLAGS= -O3 -std=c++11 -ccbin gcc-7 -g
LIBS= -lm -lstdc++
CFLAGS = -DNDIM=2

ca_main: ca_main.cu
	$(NVCC) ca_main.cu -o ca_main $(CUDAFLAGS) $(LIBS) $(CFLAGS)

example: example.cu
	$(NVCC) example.cu -o example $(CUDAFLAGS) $(LIBS) $(CFLAGS)

test: example
	./example

clean:
	rm -rf *.o example ca_main
