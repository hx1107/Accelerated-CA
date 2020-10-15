#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define USE_CUDA
//#undef USE_CUDA

#define DO_TERM_DISPLAY
//#undef DO_TERM_DISPLAY

#define BLOCK_SIZE 256

#ifndef NDIM
#define NDIM 2
#endif
#ifndef N_COLOR_BIT
#define N_COLOR_BIT 4
#endif
#ifndef CANVAS_SIZE_X
#define CANVAS_SIZE_X 100
#endif
#ifndef CANVAS_SIZE_Y
#define CANVAS_SIZE_Y 350
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

//#define idx(X, Y) (X >= 0 && Y >= 0 && ((Y * CANVAS_SIZE_X + X) < NUM_CELLS - 1) ? (Y * CANVAS_SIZE_X + X) : NUM_CELLS - 1)

typedef struct cell {
    unsigned int x : N_COLOR_BIT;
} cell;

inline size_t idx(int X, int Y)
{
    if (Y >= 0 && X >= 0 && ((Y * CANVAS_SIZE_X + X) < NUM_CELLS - 1)) {
        return (Y * CANVAS_SIZE_X + X);
    } else {
        return NUM_CELLS - 1;
    }
}
inline size_t xdi_x(size_t i)
{
    return i % CANVAS_SIZE_X;
}

inline size_t xdi_y(size_t i)
{
    return i / CANVAS_SIZE_X;
}
cell *cuda_buffer1 = NULL, *cuda_buffer2 = NULL;
cell* host_buffer;

void init()
{
    debug_print("Using %lu dimensional canvas of size %zux%zu with %d bit colors\n", NDIM, CANVAS_SIZE_X, CANVAS_SIZE_Y, N_COLOR_BIT);
    debug_print("Using two buffer each of size %lu mb, or %lu cells\n", BUFFER_SIZE / 1024 / 1024, BUFFER_SIZE / sizeof(cell));

#ifdef USE_CUDA
    if (cuda_buffer1) {
        cudaFree(cuda_buffer1);
        cuda_buffer1 = NULL;
    }
    if (cuda_buffer2) {
        cudaFree(cuda_buffer2);
        cuda_buffer2 = NULL;
    }
    cudaCalloc(&cuda_buffer1, BUFFER_SIZE, 1);
    cudaCalloc(&cuda_buffer2, BUFFER_SIZE, 1);
#else
    if (cuda_buffer1) {
        free(cuda_buffer1);
        cuda_buffer1 = NULL;
    }
    if (cuda_buffer2) {
        free(cuda_buffer2);
        cuda_buffer2 = NULL;
    }
    cuda_buffer1 = (cell*)calloc(BUFFER_SIZE, 1);
    cuda_buffer2 = (cell*)calloc(BUFFER_SIZE, 1);
#endif

    if (host_buffer) {
        free(host_buffer);
        host_buffer = NULL;
    }
    host_buffer = (cell*)calloc(1, BUFFER_SIZE);
    debug_print("Initialization done!\n");
}

inline unsigned int update_cell_bin_2d(cell& center, cell& c1, cell& c2, cell& c3, cell& c4, cell& c5, cell& c6, cell& c7, cell& c8)
{
    unsigned int sum = c1.x
        + c2.x
        + c3.x
        + c4.x
        + c5.x
        + c6.x
        + c7.x
        + c8.x;
    if (sum < 2 || sum > 3) {
        return 0;
    } else if ((center.x && (sum == 2 || sum == 3)) || (!center.x) && sum == 3) {
        return 1;
    }
    return 0;
}

__global__ void update_cell_bin_2d_CUDA()
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < CANVAS_SIZE_X * CANVAS_SIZE_Y) {
        size_t x = xdi_x(i), y = xdi_y(i);
        debug_print("Update %d: [%lu,%lu]\n", i, x, y);
    }
}

#ifndef USE_CUDA
inline
#endif
    void
    update(cell* dest, cell* origin)
{
#ifdef USE_CUDA
    //debug_print("Not Implemented!\n");
    update_cell_bin_2d_CUDA<<<ceil(CANVAS_SIZE_X * CANVAS_SIZE_Y / BLOCK_SIZE), BLOCK_SIZE>>>();
    return;
#else
    for (int x = 0; x < CANVAS_SIZE_X; x++) {
        for (int y = 0; y < CANVAS_SIZE_Y; y++) {
            /*debug_print("Update (%zu,%zu)\n",x,y);*/
            size_t i = idx(x, y);
            dest[i].x = update_cell_bin_2d(origin[i],
                origin[idx(x - 1, y - 1)],
                origin[idx(x + 1, y + 1)],
                origin[idx(x, y - 1)],
                origin[idx(x - 1, y)],
                origin[idx(x + 1, y)],
                origin[idx(x, y + 1)],
                origin[idx(x - 1, y + 1)],
                origin[idx(x + 1, y - 1)]);
        }
    }
#endif
}

inline void copy_buffer_to_host(cell* dst, cell* src)
{
#ifndef USE_CUDA
    memcpy(dst, src, BUFFER_SIZE);
#else
    cudaMemcpy(dst, src, BUFFER_SIZE, cudaMemcpyDeviceToHost);
#endif
}
inline void copy_buffer_to_device(cell* dst, cell* src)
{
#ifndef USE_CUDA
    memcpy(dst, src, BUFFER_SIZE);
#else
    cudaMemcpy(dst, src, BUFFER_SIZE, cudaMemcpyHostToDevice);
#endif
}

inline void print_buffer(cell* src)
{
#ifdef DO_TERM_DISPLAY
    fprintf(stdout, "---------------------Iteration-------------------------\n");
    for (int x = 0; x < CANVAS_SIZE_X; x++) {
        for (int y = 0; y < CANVAS_SIZE_Y; y++) {
            //debug_print("(%zu, %zu) -> %zu\n", x, y, idx(x, y));
            fprintf(stdout, src[idx(x, y)].x ? " " : "█");
        }
        fprintf(stdout, "|\n");
    }
#endif
}

int main(void)
{
    debug_print("-------------CA Running-----------\n");
    init();
    for (int i = CANVAS_SIZE_X * 3 / 7; i < CANVAS_SIZE_X * 4 / 7; i++) {
        host_buffer[idx(i, i)].x = 1;
        host_buffer[idx(i, i + 1)].x = 1;
        host_buffer[idx(i, i - 1)].x = 1;
        host_buffer[idx(i, i - 5)].x = 1;
        host_buffer[idx(i - 1, i - 6)].x = 1;
        host_buffer[idx(i, i - 6)].x = 1;
    }
    print_buffer(host_buffer);
    copy_buffer_to_device(cuda_buffer1, host_buffer);

    int iteration = 20000;
#ifdef DO_TERM_DISPLAY
    int delay = 30000;
#else
    int delay = 0;
#endif
    for (int i = 0; i < iteration / 2; i++) {
        update(cuda_buffer2, cuda_buffer1);
        copy_buffer_to_host(host_buffer, cuda_buffer2);
        print_buffer(host_buffer);
        usleep(delay);

        update(cuda_buffer1, cuda_buffer2);
        copy_buffer_to_host(host_buffer, cuda_buffer1);
        print_buffer(host_buffer);
        usleep(delay);
    }

    /*
       cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i] - 4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
     */
}
/*__global__ void saxpy(int n, float a, float* x, float* y)
  {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  y[i] = a * x[i] + y[i];
  }*/
