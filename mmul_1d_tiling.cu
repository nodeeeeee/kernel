

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

using std::vector;
using std::generate;
using std::cout;

__global__ void mmul(int N, int M, int K, const int *a, const int *b, int *c) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x; //64   col
    const int y = blockIdx.y * blockDim.y + threadIdx.y; //8  row
    const int TM = 8; // tile size
    int result[TM] = {0};
    if (x < M && y < N) {
        const int posx = threadIdx.x;
        const int posy = threadIdx.y;
        const int BLOCKSIZE = 64;
        __shared__ int As[BLOCKSIZE * TM];  // 64x8
        __shared__ int Bs[TM * BLOCKSIZE];  // 8x64
        for (int blkIdx = 0; blkIdx < K; blkIdx += TM) {
            // each thread loads its corresponding piece into As and Bs
            int globalRow = blockIdx.y * 64 + posx; // we have to draw a picture to better show the coordination trasition.
            As[posx * TM + posy] = a[globalRow * K + blkIdx + posy];     // N * K
            Bs[posy * BLOCKSIZE + posx] = b[blkIdx * M + posy * M + x]; // K * M
            __syncthreads();   //syncing among all blocks (seems not necessary?)
            // calculate

            for (int dotIdx = 0; dotIdx < TM; dotIdx++) {
                int b_val = Bs[dotIdx * BLOCKSIZE + posx];
                for (int resIdx = 0; resIdx < TM; resIdx++) {
                    result[resIdx] += b_val * As[(posy * 8 + resIdx) * 8 + dotIdx]; 
                }
            }
            __syncthreads();
        }
        
        for (int i = 0; i < TM; i++) {
            c[(y * 8 + i) * M + x] = result[i];
        }
    }
}

int CEIL_DIV(int a, int b) {
    return (a + b - 1) / b;
}

bool verify(int N, int M, int K, vector<int> &a, vector<int> &b, vector<int> &c) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int tmp = 0;
            for (int k = 0; k < K; k++) {
                tmp += a[i * K + k] * b[k * M + j];
            }
            if (c[i * M + j] != tmp) {
                printf("failed at %d, %d\n", i, j);
                printf("should be %d, now is %d\n", tmp, c[i * M + j]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int N = 1<<9;
    int M = 1<<11;
    int K = 1<<10;
    size_t a_bytes = N * K * sizeof(int);
    size_t b_bytes = K * M * sizeof(int);
    size_t c_bytes = N * M * sizeof(int);
    vector<int> h_a(N * K);
    vector<int> h_b(K * M);
    vector<int> h_c(N * M);
    generate(h_a.begin(), h_a.end(), [](){return rand() % 1000; });
    generate(h_b.begin(), h_b.end(), [](){return rand() % 1000; });

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, c_bytes);

    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));   //dim3(x, y, z)    x: col   y: row
    dim3 blockDim(64, 8); //64 col, 8 row

    mmul<<<gridDim, blockDim>>>(N, M, K, d_a, d_b, d_c); // grid should be N * M

    cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost);
    cout << "MMUL DONE\n";
    assert(verify(N, M, K, h_a, h_b, h_c) == true);
    
    cout << "COMPLETED SUCCESSFULLY";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

}