#include "config.h"
#include "load_rle.h"
#include "render.h"
#include <stdio.h>
#include <time.h>
#include <unistd.h>

static cell *cuda_buffer1 = NULL, *cuda_buffer2 = NULL;
cell* host_buffer = NULL;

__host__ __device__ inline size_t idx(int X, int Y)
{
    if (Y >= 0 && X >= 0 && ((Y * CANVAS_SIZE_X + X) < NUM_CELLS - 1)) {
        return (Y * CANVAS_SIZE_X + X);
    } else {
        return NUM_CELLS - 1;
    }
}
//#define idx(X, Y) \
    //(((int)Y >= 0 && (int)X >= 0 && ((Y * CANVAS_SIZE_X + X) < NUM_CELLS - 1)) ? (Y * CANVAS_SIZE_X + X) : NUM_CELLS - 1)
__host__ __device__ inline size_t xdi_x(size_t i)
{
    return i % CANVAS_SIZE_X;
}

__host__ __device__ inline size_t xdi_y(size_t i)
{
    return i / CANVAS_SIZE_X;
}

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

__host__ __device__ inline unsigned int update_cell_bin_2d(cell& center, cell& c1, cell& c2, cell& c3, cell& c4, cell& c5, cell& c6, cell& c7, cell& c8)
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

__global__ void update_cell_bin_2d_CUDA(cell* dest, cell* origin)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < CANVAS_SIZE_X * CANVAS_SIZE_Y) {
        size_t x = xdi_x(i), y = xdi_y(i);
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

#ifndef USE_CUDA
inline
#endif
    void
    update(cell* dest, cell* origin)
{
#ifdef USE_CUDA
    update_cell_bin_2d_CUDA<<<ceil(CANVAS_SIZE_X * CANVAS_SIZE_Y / BLOCK_SIZE), BLOCK_SIZE>>>(dest, origin);
    return;
#else
    for (int x = 0; x < CANVAS_SIZE_X; x++) {
        for (int y = 0; y < CANVAS_SIZE_Y; y++) {
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
    //for (int i = CANVAS_SIZE_X * 3 / 7; i < CANVAS_SIZE_X * 4 / 7; i++) {
    //host_buffer[idx(i, i)].x = 1;
    //host_buffer[idx(i, i + 1)].x = 1;
    //host_buffer[idx(i, i - 1)].x = 1;
    //host_buffer[idx(i, i - 5)].x = 1;
    //host_buffer[idx(i - 1, i - 6)].x = 1;
    //host_buffer[idx(i, i - 6)].x = 1;
    //}
    //load_rle_file(host_buffer, "./saves/glider.rle", .5, .5);
    //load_rle_file(host_buffer, "./saves/Gosper glider gun.rle", .5, .5);
    //load_rle_file(host_buffer, "./saves/rats_synth.rle", .5, .5);
    load_rle_file(host_buffer, "./saves/fullutm.rle", 0, 0);
    print_buffer(host_buffer);
    copy_buffer_to_device(cuda_buffer1, host_buffer);

    render_init();

    int iteration = 80000;
#ifdef DO_TERM_DISPLAY
    int delay = 30000;
#else
    int delay = 30000;
    //int delay = 0;
#endif
    //while (0 == display_ready)
    //; //wait for display
    debug_print("Display is now ready\n");
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
}
