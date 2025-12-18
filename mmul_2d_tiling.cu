#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

using std::vector;
using std::generate;
using std::cout;

__global__ void mmul(int N, int M, int K, const int *a, const int *b, int *c) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x; //32   col
    const int y = blockIdx.y * blockDim.y + threadIdx.y; //32  row
    //tile
    const int TM = 2; 
    const int TN = 2;
    //block
    const int BN = 64, BM = 64, BK = 64;
    int result[TM][TN] = {0};
    if (x < M && y < N) {
        const int posx = threadIdx.x;
        const int posy = threadIdx.y;
        __shared__ int As[BN * BK];  // 64x64
        __shared__ int Bs[BK * BM];  // 64x64
        int regA[TN]; // a col
        int regB[TM]; // a row
        for (int blkIdx = 0; blkIdx < K; blkIdx += BK) {
            // each thread loads its corresponding piece into As and Bs
            int aStartPos = blockIdx.y * BN * K + blkIdx; // we have to draw a picture to better show the coordination trasition.
            int bStartPos = blkIdx * M + blockIdx.x * BM;
            int offset = 4 * (posy * blockDim.x + posx);
            for (int tag = 0; tag < TM * TN; tag++) {
                As[offset + tag] = a[aStartPos + offset / BK * K + offset % BK + tag];     // N * K
                Bs[offset + tag] = b[bStartPos + offset / BM * M + offset % BM + tag]; // K * M
            }
            __syncthreads();   //syncing among all blocks (seems not necessary?)
            

            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                for (int i = 0; i < TN; i++) {
                    regA[i] = As[(posy * TN + i) * BK + dotIdx]; // be careful       posy * TN + i!!!!!!!!   stuck for hours
                }
                for (int i = 0; i < TM; i++) {
                    regB[i] = Bs[dotIdx * BM + posx * TM + i];
                }
                for (int i = 0; i < TN; i++) {
                    for (int j = 0; j < TM; j++) {
                        result[i][j] += regA[i] * regB[j];
                    }
                }
            }
            __syncthreads();
        }
        
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
                c[(y * TM + i) * M + x * TN + j] = result[i][j];
            }
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
    int N = 1<<8;
    int M = 1<<10;
    int K = 1<<9;
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
    dim3 blockDim(32, 32); //32 col, 32 row

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