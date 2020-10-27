#pragma once
#ifndef CONFIG_H

#define DEBUG

#define USE_CUDA

//#define DO_TERM_DISPLAY

#define BLOCK_SIZE 512

#ifndef NDIM
#define NDIM 2
#endif
#ifndef N_COLOR_BIT
#define N_COLOR_BIT sizeof(float)
#endif
#ifndef CANVAS_SIZE_X
//#define CANVAS_SIZE_X 100
#define CANVAS_SIZE_X 500
#endif
#ifndef CANVAS_SIZE_Y
//#define CANVAS_SIZE_Y 375
#define CANVAS_SIZE_Y 500
#endif
#ifndef NUM_RULE
#define NUM_RULE 10
#endif
#ifndef BUFFER_SIZE
#define BUFFER_SIZE ((CANVAS_SIZE_X * CANVAS_SIZE_Y + 1) * sizeof(cell))
#define NUM_CELLS (BUFFER_SIZE / sizeof(cell))
#endif
#ifdef DEBUG
#define debug_print(...)     \
    do {                     \
        printf(__VA_ARGS__); \
    } while (0)
#else
#define debug_print(...) \
    do {                 \
    } while (0)
#endif
#define cudaCalloc(A, B, C)                                  \
    do {                                                     \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B * C); \
        if (__cudaCalloc_err == cudaSuccess)                 \
            cudaMemset(*A, 0, B* C);                         \
    } while (0)
struct cell;
static cell* host_buffer = nullptr;
#endif