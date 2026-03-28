#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Tri par merge sort bottom-up en shared memory.
// On suppose d puissance de 2, d <= 1024.
__global__ void sortSmallBatch_k(int* M, int d, int N) {
    extern __shared__ int smem[];

    int Qt   = threadIdx.x / d;
    int tidx = threadIdx.x - Qt * d;
    int gbx  = Qt + blockIdx.x * (blockDim.x / d);

    // 2 buffers ping-pong par groupe dans la shared memory
    int* src = smem + Qt * 2 * d;
    int* dst = src + d;

    src[tidx] = (gbx < N) ? M[gbx * d + tidx] : 0;
    __syncthreads();

    for (int w = 1; w < d; w *= 2) {
        int local = tidx % (2 * w);
        int base  = tidx - local;
        int sA = w, sB = w;

        int* A = src + base;
        int* B = src + base + sA;
        int i = local;

        int Kx, Ky, Px, Py;
        if (i > sA) {
            Kx = i - sA; Ky = sA;
            Px = sA;     Py = i - sA;
        } else {
            Kx = 0; Ky = i;
            Px = i; Py = 0;
        }

        while (1) {
            int offset = abs(Ky - Py) / 2;
            int Qx = Kx + offset;
            int Qy = Ky - offset;

            if (Qy >= 0 && Qx <= sB &&
                (Qy == sA || Qx == 0 || A[Qy] > B[Qx - 1])) {

                if (Qx == sB || Qy == 0 || A[Qy - 1] <= B[Qx]) {
                    dst[tidx] = (Qy < sA && (Qx == sB || A[Qy] <= B[Qx]))
                               ? A[Qy] : B[Qx];
                    break;
                } else {
                    Kx = Qx + 1;
                    Ky = Qy - 1;
                }
            } else {
                Px = Qx - 1;
                Py = Qy + 1;
            }
        }

        // __syncthreads() doit être appelé par tous les threads du bloc,
        // y compris ceux hors-borne (gbx >= N), d'où l'absence de return anticipé
        __syncthreads();
        int* tmp = src; src = dst; dst = tmp;
    }

    if (gbx < N)
        M[gbx * d + tidx] = src[tidx];
}

int main() {

    printf("=== TEST DE CORRECTION (N=3, d=8) ===\n");
    {
        int N = 3, d = 8;
        int h_M[] = {
            5, 3, 8, 1, 9, 2, 7, 4,
            10, 6, 3, 8, 1, 5, 2, 9,
            7, 7, 3, 1, 4, 6, 2, 5
        };

        int* d_M;
        cudaMalloc(&d_M, N * d * sizeof(int));
        cudaMemcpy(d_M, h_M, N * d * sizeof(int), cudaMemcpyHostToDevice);

        int tpb = (1024 / d) * d;
        int numBlocks = (N + tpb/d - 1) / (tpb/d);
        sortSmallBatch_k<<<numBlocks, tpb, 2*tpb*sizeof(int)>>>(d_M, d, N);

        cudaMemcpy(h_M, d_M, N * d * sizeof(int), cudaMemcpyDeviceToHost);
        for (int p = 0; p < N; p++) {
            printf("M%d = ", p + 1);
            for (int j = 0; j < d; j++) printf("%d ", h_M[p*d+j]);
            printf("\n");
        }
        // Attendu :
        // M1 = 1 2 3 4 5 7 8 9
        // M2 = 1 2 3 5 6 8 9 10
        // M3 = 1 2 3 4 5 6 7 7
        cudaFree(d_M);
    }

    printf("\n=== TEMPS EN FONCTION DE d (totalElems=2^24) ===\n");
    printf("%-8s %-12s %-10s %-10s\n", "d", "N", "tpb", "temps(ms)");

    int ds[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int totalElems = 1 << 24;

    for (int k = 0; k < (int)(sizeof(ds)/sizeof(ds[0])); k++) {
        int d = ds[k];
        int N = totalElems / d;
        int tpb = (1024 / d) * d;
        int numBlocks = (N + tpb/d - 1) / (tpb/d);

        int* d_M;
        cudaMalloc(&d_M, (long)N * d * sizeof(int));
        cudaMemset(d_M, 0, (long)N * d * sizeof(int));

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        sortSmallBatch_k<<<numBlocks, tpb, 2*tpb*sizeof(int)>>>(d_M, d, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);

        printf("%-8d %-12d %-10d %.3f\n", d, N, tpb, ms);

        cudaFree(d_M);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    return 0;
}



