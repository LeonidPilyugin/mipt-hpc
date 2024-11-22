#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// element of matrix
#define INDEX(r, c, rows, columns) ((c) + (r) * (columns))
// element of transposed matrix
#define TINDEX(r, c, rows, columns) ((r) + (c) * (rows))


// keep dimensions multiple to BLOCK
#define X 8192
#define Y 8192
#define Z 8192

#define BLOCK 32

#define BX blockIdx.x
#define BY blockIdx.y
#define TX threadIdx.x
#define TY threadIdx.y

/*
 * left -- left matrix
 * right -- transposed right matrix
 */
__global__ void global_matrix_multiply(double * left, double * right, int3 size, double * result) {
    int rr = BX * BLOCK + TX;
    int rc = BY * BLOCK + TY;
    double sum = 0.0;
    for (int i = 0; i < size.x; i++) {
        sum += left[INDEX(rr, i, size.y, size.x)] * right[INDEX(i, rc, size.x, size.z)];
    }
    result[INDEX(rr, rc, size.y, size.z)] = sum;
}

__global__ void shared_matrix_multiply(double * left, double * right, int3 size, double * result) {
    __shared__ double s_left[BLOCK][BLOCK];
    __shared__ double s_right[BLOCK][BLOCK];

    int rr = BX * BLOCK + TX;
    int rc = BY * BLOCK + TY;
    double sum = 0.0;

    for (int i = 0; i < size.x; i += BLOCK) {
        // load chunks to shared memory
        s_left[TY][TX] = left[INDEX(rr, TY + i, size.y, size.x)];
        s_right[TY][TX] = right[INDEX(TX + i, rc, size.x, size.z)];
        __syncthreads();

        // compute sum
        for (int j = 0; j < BLOCK; j++)
            sum += s_left[j][TX] * s_right[TY][j];

        __syncthreads();
    }

    result[INDEX(rr, rc, size.y, size.z)] = sum;
}

double * generate_matrix(int rows, int columns, int seed=0, double min=0.0, double max=5.0) {
    double * result;
    #ifdef PINNED
    cudaMallocHost((void **) &result, rows * columns * sizeof(double));
    #else
    result = (double *) malloc(rows * columns * sizeof(double));
    #endif

    srand(seed);

    for (int i = 0; i < rows * columns; i++)
        result[i] = min + rand() * (max - min) / ((double) RAND_MAX);

    return result;
}

void print_matrix(double * m, int rows, int columns, FILE * fp) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < columns; column++) {
            fprintf(fp, "%lf ", m[INDEX(row, column, rows, columns)]);
        }
        fprintf(fp, "\n");
    }
}

void destroy_matrix(double * m) {
    #ifdef PINNED
    cudaFreeHost(m);
    #else
    free(m);
    #endif
}

#ifdef SHARED
#define DEVICE_MULTIPLY shared_matrix_multiply
#else
#define DEVICE_MULTIPLY global_matrix_multiply
#endif

#define FOREACH_STREAM(n) for (int i = 0; i < (n); i++)

//#define STREAMS 2

void host_multiply(double * left, double * right, int3 size, double * result) {
    double * dl, * dr, * dres;
    cudaMalloc((void **) &dl, size.x * size.y * sizeof(double));
    cudaMalloc((void **) &dr, size.x * size.z * sizeof(double));
    cudaMalloc((void **) &dres, size.y * size.z * sizeof(double));

    cudaMemcpy(dr, right, sizeof(double) * size.x * size.z, cudaMemcpyHostToDevice);

    #ifdef STREAMS
    cudaStream_t streams[STREAMS];
    FOREACH_STREAM(STREAMS) cudaStreamCreate(streams + i);

    int3 new_size = make_int3(size.x, size.y / STREAMS, size.z);
    
    FOREACH_STREAM(STREAMS) cudaMemcpyAsync(dl + i * size.x * new_size.y, left + i * size.x * new_size.y, size.x * new_size.y * sizeof(double), cudaMemcpyHostToDevice, streams[i]);

    FOREACH_STREAM(STREAMS) 
    DEVICE_MULTIPLY<<<dim3(new_size.y / BLOCK, size.z / BLOCK), dim3(BLOCK, BLOCK), 0, streams[i]>>>(dl + i * size.x * new_size.y, dr, new_size, dres + i * size.z * new_size.y);

    FOREACH_STREAM(STREAMS) cudaMemcpyAsync(result + i * size.z * new_size.y, dres + i * size.z * new_size.y, size.z * new_size.y * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);

    cudaDeviceSynchronize();

    FOREACH_STREAM(STREAMS) cudaStreamDestroy(streams[i]);
    #else
    cudaMemcpy(dl, left, sizeof(double) * size.x * size.y, cudaMemcpyHostToDevice);
    DEVICE_MULTIPLY<<<dim3(size.y / BLOCK, size.z / BLOCK), dim3(BLOCK, BLOCK)>>>(dl, dr, size, dres);

    cudaMemcpy(result, dres, sizeof(double) * size.z * size.y, cudaMemcpyDeviceToHost);
    #endif

    cudaFree(dl);
    cudaFree(dr);
    cudaFree(dres);
}

int main(int argc, char * argv[]) {
    int3 size = make_int3(X, Y, Z);
    double * A = generate_matrix(size.y, size.x, 1);
    double * B = generate_matrix(size.x, size.z, 2);
    double * C = generate_matrix(size.y, size.z, 3);

    clock_t time = clock();
    host_multiply(A, B, size, C);
    time = clock() - time;

    printf("Elapsed time: %lf ms\n", ((double) time) / CLOCKS_PER_SEC * 1000.0);

    FILE * f = fopen("A.matrix", "w");
    print_matrix(A, size.y, size.x, f);
    fclose(f);

    f = fopen("B.matrix", "w");
    print_matrix(B, size.x, size.z, f);
    fclose(f);

    f = fopen("C.matrix", "w");
    print_matrix(C, size.y, size.z, f);
    fclose(f);

    destroy_matrix(A);
    destroy_matrix(B);
    destroy_matrix(C);

    return 0;
}

