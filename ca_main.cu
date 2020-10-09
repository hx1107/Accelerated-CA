#include <stdio.h>

#ifndef NDIM
#define NDIM 2
#endif
#ifndef N_COLOR_BIT
#define N_COLOR_BIT 1
#endif
#ifndef CANVAS_SIZE_X
#define CANVAS_SIZE_X 100
#endif
#ifndef CANVAS_SIZE_Y
#define CANVAS_SIZE_Y 100
#endif
#ifndef NUM_RULE
#define NUM_RULE 10
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
#define cudaCalloc(A, B, C) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B*C); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, B*C); \
    } while (0)

#define idx(X,Y) ((X*CANVAS_SIZE_Y+Y))

typedef struct cell{
    unsigned int x:N_COLOR_BIT;
}cell;

cell *cuda_buffer1 = NULL, *cuda_buffer2 = NULL;
cell* host_buffer;

void init()
{
    debug_print("Using %lu dimensional canvas of size %zux%zu with %d bit colors\n", NDIM, CANVAS_SIZE_X, CANVAS_SIZE_Y, N_COLOR_BIT);
    size_t buffer_size = CANVAS_SIZE_X*CANVAS_SIZE_Y;
    buffer_size *= sizeof(cell);
    debug_print("Using two buffer each of size %lu\n", buffer_size);
    if(cuda_buffer1){cudaFree(cuda_buffer1);cuda_buffer1=NULL;}
    if(cuda_buffer2){cudaFree(cuda_buffer2);cuda_buffer1=NULL;}
    if(host_buffer){free(host_buffer);host_buffer=NULL;}
    cudaCalloc(&cuda_buffer1,buffer_size,1);
    cudaCalloc(&cuda_buffer2,buffer_size,1);
    host_buffer=(cell*)calloc(1,buffer_size);

}
void update(){

}
int main(void)
{
    debug_print("-------------CA Running-----------\n");
    init();
    host_buffer[idx(0,50)].x=1;

    int iteration = 100;
    for(int i=0;i<iteration/2;i++){
        update();
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
