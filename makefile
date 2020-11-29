CC= g++
NVCC= nvcc
CUDAFLAGS= -ccbin gcc-7 -g
LIBS= -lm -lstdc++ -lGL -lSDL2 -lGLEW -lGLU -lglut
CFLAGS= -DNDIM=2 -O3 -std=c++11 -g
OBJS= ca_main.o render.o

all: ca_main

ca_main: ca_main.cu render.o render.h load_rle.o load_rle.h
	$(NVCC) -o ca_main ca_main.cu render.o load_rle.o $(CUDAFLAGS) $(LIBS) $(CFLAGS)

render.o: render.cpp render.h
	$(CC) -c render.cpp $(LIBS) $(CFLAGS)

load_rle.o: load_rle.cpp load_rle.h
	$(CC) -c load_rle.cpp $(LIBS) $(CFLAGS)

example: example.cu
	$(NVCC) example.cu -o example $(CUDAFLAGS) $(LIBS) $(CFLAGS)

test: example
	./example

clean:
	rm -rf *.o example ca_main
