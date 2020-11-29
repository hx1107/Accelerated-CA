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
#define N_COLOR_BIT (sizeof(float))
#endif
#ifndef CANVAS_SIZE_X
//#define CANVAS_SIZE_X 128
//#define CANVAS_SIZE_X 500
//#define CANVAS_SIZE_X 1920
#define CANVAS_SIZE_X (1920 * 2)
//#define CANVAS_SIZE_X (ssize_t)99551
#endif
#ifndef CANVAS_SIZE_Y
//#define CANVAS_SIZE_Y 128
//#define CANVAS_SIZE_Y 500
//#define CANVAS_SIZE_Y 1080
#define CANVAS_SIZE_Y (1080 * 2)
//#define CANVAS_SIZE_Y (ssize_t)99343
#endif
#ifndef NUM_RULE
#define NUM_RULE 10
#endif
#ifndef BUFFER_SIZE
#define BUFFER_SIZE ((CANVAS_SIZE_X * CANVAS_SIZE_Y + 1) * sizeof(cell))
#define NUM_CELLS (BUFFER_SIZE / sizeof(cell))
#endif

#define RATIO_CELL_TO_PIXEL 2
#define RENDER_WINDOW_WIDTH (CANVAS_SIZE_X * RATIO_CELL_TO_PIXEL)
#define RENDER_WINDOW_HEIGHT (CANVAS_SIZE_Y * RATIO_CELL_TO_PIXEL)
#define RENDER_WIDTH 0.95

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
#define COUNT_OF(x) ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))
#define cudaCalloc(A, B, C)                                  \
    do {                                                     \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B * C); \
        if (__cudaCalloc_err == cudaSuccess)                 \
            cudaMemset(*A, 0, B* C);                         \
    } while (0)
struct cell;
typedef struct cell {
    //unsigned int x : N_COLOR_BIT;
    float x;
} cell;
extern cell* host_buffer;
extern int display_ready;
#endif
