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
__global__ void mmul(int N, int M, int K, int TN, int TM, int BN, int BM, int BK, int WNITER, int WMITER, const T *a, const T *b, T *c) {
    using T4 = typename VecType<T>::type;
    const int x = blockIdx.x * blockDim.x + threadIdx.x; //32   col
    const int y = blockIdx.y * blockDim.y + threadIdx.y; //32  row
    const int WM = BM / WNITER;
    const int WN = BN / WMITER;

    T result[16][16] = {0};
    if (x < M && y < N) {
        const int posx = threadIdx.x;
        const int posy = threadIdx.y;
        extern __shared__ T shmem[];
        T *As = shmem;
        T *Bs = &shmem[BN * BK];
        T regA[16]; // a col
        T regB[16]; // a row

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
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            T tmp = 0;
            for (int k = 0; k < K; k++) {
                tmp += a[i * K + k] * b[k * M + j];
            }
            if (c[i * M + j] != tmp) {
                printf("failed at %d, %d\n", i, j);
                printf("should be %f, now is %f\n", tmp, c[i * M + j]);
                return false;
            }
        }
    }
    return true;
}

template<typename T>
float run(int N, int M, int K, int  TN, int TM, int BN, int BM, int BK, int WNITER, int WMITER, T* d_a, T* d_b, T* d_c, vector<T> h_a, vector<T> h_b, vector<T> h_c) {
    size_t c_bytes = N * M * sizeof(float);
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));   //dim3(x, y, z)    x: col   y: row
    dim3 blockDim(BM / TM / WMITER, BN / TN / WNITER); //32 col, 32 row
    int shmem_size = (BN * BK + BK * BM) * sizeof(T);
    cudaMemset(d_c, 0, c_bytes);
    if (shmem_size > 49152) {
        return 1000000;
    }
    //warm up run
    printf("warm up\n");
    mmul<float><<<gridDim, blockDim, shmem_size>>>(N, M, K, TN, TM, BN, BM, BK, WNITER, WMITER, d_a, d_b, d_c); // grid should be N * M
    mmul<float><<<gridDim, blockDim, shmem_size>>>(N, M, K, TN, TM, BN, BM, BK, WNITER, WMITER, d_a, d_b, d_c); // grid should be N * M

    printf("warm up done\n");
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mmul<float><<<gridDim, blockDim, shmem_size>>>(N, M, K, TN, TM, BN, BM, BK, WNITER, WMITER, d_a, d_b, d_c); // grid should be N * M
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // 7. 销毁事件资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost);
    cout << "MMUL DONE\n";
    // printf("TN: %d, TM: %d, BN: %d, BM: %d, BK: %d\n", TN, TM, BN, BM, BK);
    // assert(verify<float>(N, M, K, h_a, h_b, h_c) == true);

    // cout << "COMPLETED SUCCESSFULLY\n";


    return milliseconds;
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

    // printf("START TEST:\n---------------\n");
    // run<float>(N, M, K, 2, 2, 32, 32, 32, d_a, d_b, d_c, h_a, h_b, h_c);
    // printf("TEST DONE\n");

    float min_time = 100000;
    int TNs = 0, TMs = 0, BKs = 0, WNITERs = 0, WMITERs = 0, BNs = 0, BMs = 0;

    for (int TM = 2; TM <= 16; TM *= 2) {
        int TN = TM;
        for (int BK = 8; BK <= min(128, TM * 32); BK *= 2) {
            for (int WNITER = 2; WNITER <= 16 / TN; WNITER *= 2) {
                int WMITER = WNITER;
                for (int BN = 32; BN <= min(128, BK * TN * TM * WNITER * WMITER / 4); BN *= 2) {
                    int BM = BN;
                    float res = run<float>(N, M, K, TN, TM, BN, BM, BK, WNITER, WMITER, d_a, d_b, d_c, h_a, h_b, h_c);
                    if (min_time > res) {
                        min_time = res;
                        TNs = TN;
                        TMs = TM;
                        BKs = BK;
                        WNITERs = WNITER;
                        WMITERs = WMITER;
                        BNs = BN;
                        BMs = BM;
                    }

                }
            }
        }
    }
    printf("min time: %f\nTN: %d, TM: %d, BK: %d, WNITER: %d, WMITER: %d, BN: %d, BM: %d\n", min_time, TNs, TMs, BKs, WNITERs, WMITERs, BNs, BMs);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}


// TN: 4 TM: 4 BN: 64 BM: 64 BK: 16