/** this version added vectorization in GMEM->SMEM and reg->GMEM. 
 * did not transpose the matrix
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

using std::vector;
using std::generate;
using std::cout;

template<typename T> struct VecType; //occupation in case of undefined/unspecialized type T;
template<> struct VecType<float> {using type = float4; };
template<> struct VecType<int> {using type = int4; };

template<typename T>
__global__ void mmul(int N, int M, int K, const T *a, const T *b, T *c) {
    using T4 = typename VecType<T>::type;
    const int x = blockIdx.x * blockDim.x + threadIdx.x; //32   col
    const int y = blockIdx.y * blockDim.y + threadIdx.y; //32  row
    //tile
    const int TM = 4; 
    const int TN = 4;
    const int BN = 64, BM = 64, BK = 8;
    const int WM = BM / 2;
    const int WN = BN / 2;//split into 4 warps
    const int WMITER = 2; 
    const int WNITER = 2;//each thread handles 4 TN*TM area
    //64 threads
    //block

    T result[TN * WNITER][TM * WMITER] = {0};
    if (x < M && y < N) {
        const int posx = threadIdx.x;
        const int posy = threadIdx.y;
        __shared__ T As[BN * BK];  // 64x64
        __shared__ T Bs[BK * BM];  // 64x64
        T regA[TN * WNITER]; // a col
        T regB[TM * WMITER]; // a row

        for (int blkIdx = 0; blkIdx < K; blkIdx += BK) {
            // each thread loads its corresponding piece into As and Bs
            int aStartPos = blockIdx.y * BN * K + blkIdx; // we have to draw a picture to better show the coordination trasition.
            int bStartPos = blkIdx * M + blockIdx.x * BM;
            for (int i = 0; i < WMITER * WNITER * TN * TM * BK / BM; i += 4) {
                int offset = WMITER * WNITER * TM * TN * BK / BM * (posy * blockDim.x + posx) + i;
                reinterpret_cast<T4*>(&As[offset])[0] = reinterpret_cast<const T4*>(&a[aStartPos + offset / BK * K + offset % BK])[0];
                reinterpret_cast<T4*>(&Bs[offset])[0] = reinterpret_cast<const T4*>(&b[bStartPos + offset / BM * M + offset % BM])[0];
            }
            __syncthreads();   //syncing among all blocks (seems not necessary?)
            
            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
                    for (int i = 0; i < TM; i++) {
                        regA[wSubRowIdx * TM + i] = As[(wSubRowIdx * WN + posy * TN + i) * BK + dotIdx];
                    }
                }

                for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
                    for (int i = 0; i < TN; i++) {
                        regB[wSubColIdx * TN + i] = Bs[dotIdx * BM + wSubColIdx * WM + posx * TM + i];
                    }
                }

                // for (int i = 0; i < TN; i++) {
                //     //broadcast in a warp already
                //     //每一个周期只有一个warp在运行，能够broadcast就完全不需要考虑bank conflict了，所以根本没必要transpose或者拉伸矩阵。
                //     regA[i] = As[(posy * TN + i) * BK + dotIdx]; // be careful       posy * TN + i!!!!!!!!   stuck for hours
                // }
                // for (int i = 0; i < TM; i++) {
                //     regB[i] = Bs[dotIdx * BM + posx * TM + i];
                // }
                for (int i = 0; i < TN * WNITER; i++) {
                    for (int j = 0; j < TM * WMITER; j++) {
                        result[i][j] += regA[i] * regB[j];
                    }
                }
            }
            __syncthreads();
        }
        
        for (int i = 0; i < TN; i++) {
            for (int j = 0; j < TM; j++) {
                for (int k = 0; k < WNITER; k++) {
                    for (int l = 0; l < WMITER; l++) {
                        c[(blockIdx.y * BN + posy * TN + i + k * WN) * M + blockIdx.x * BM + posx * TM + j + l * WM] = result[i + k * TN][j + l * TM];
                    }
                }
            }
        }
    }
}

int CEIL_DIV(int a, int b) {
    return (a + b - 1) / b;
}

template<typename T>
bool verify(int N, int M, int K, vector<T> &a, vector<T> &b, vector<T> &c) {
    bool flag = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            T tmp = 0;
            for (int k = 0; k < K; k++) {
                tmp += a[i * K + k] * b[k * M + j];
            }
            if (c[i * M + j] != tmp) {
                printf("failed at %d, %d\n", i, j);
                printf("should be %f, now is %f\n", tmp, c[i * M + j]);
                // flag = false;
                // if (c[i * M + j] == 0) {
                //     printf("0\n");
                // }
                return false;
            }
        }
    }
    return flag;
}

int main() {
    int N = 1<<9;
    int M = 1<<11;
    int K = 1<<10;
    size_t a_bytes = N * K * sizeof(float);
    size_t b_bytes = K * M * sizeof(float);
    size_t c_bytes = N * M * sizeof(float);
    vector<float> h_a(N * K);
    vector<float> h_b(K * M);
    vector<float> h_c(N * M);
    generate(h_a.begin(), h_a.end(), [](){return rand() % 1000; });
    generate(h_b.begin(), h_b.end(), [](){return rand() % 1000; });

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    cudaMemcpy(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, c_bytes);

    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));   //dim3(x, y, z)    x: col   y: row
    dim3 blockDim(8, 8); //32 col, 32 row

    mmul<float><<<gridDim, blockDim>>>(N, M, K, d_a, d_b, d_c); // grid should be N * M

    cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost);
    cout << "MMUL DONE\n";
    assert(verify(N, M, K, h_a, h_b, h_c) == true);
    
    cout << "COMPLETED SUCCESSFULLY";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

}