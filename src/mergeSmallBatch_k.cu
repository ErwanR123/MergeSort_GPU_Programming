#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib> 
#include "timer.h"

using std::generate;

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Renamed macro to avoid naming conflicts with the function
#define CUDA_CHECK(error) (testCUDA(error, __FILE__ , __LINE__))

void mergeSmallBatch_CPU(int* A, int* B,
                         int d, int N,
                         int* M){

    for (int batch = 0; batch < N; batch++){
        int* myA = A + batch * d;
        int* myB = B + batch * d;
        int* myM = M + batch * d;

        int i = 0;
        int j = 0;

        while (i+j < 2 * d)
        {
            if (i >= d){
                myM[i+j] = myB[j];
                j += 1;
            }
            else if (j >= d)
            {
                myM[i+j] = myA[i];
                i += 1;
            }
            else if (myA[i] < myB[j])
            {
                myM[i+j] = myA[i];
                i += 1;
            }
            else
            {
                myM[i+j] = myB[j];
                j += 1;
            }
        }    
    }
}

__global__ void mergeSmallBatch_k(int* A, int* B, int* M,
                                   int n_A, int n_B, int d, int N) {
    // Local fusion id among same block 
    int Qt   = threadIdx.x / d;
    // Local thread id among same fusion
    int tidx = threadIdx.x - (Qt * d);
    // Global fusion id among all the pairs
    int gbx  = Qt + blockIdx.x * (blockDim.x / d);

    if (gbx >= N) return;

    // Pointeurs vers la paire gbx dans les tableaux batch
    int* myA = A + gbx * n_A;
    int* myB = B + gbx * n_B;
    int* myM = M + gbx * d;

    // Same merge path and same logic as mergeSmall_k
    int i = tidx;

    int Kx, Ky, Px, Py;
    if (i > n_A) {
        Kx = i - n_A;  
        Ky = n_A;

        Px = n_A;
        Py = i - n_A;

    } else {
        Kx = 0;
        Ky = i;

        Px = i;
        Py = 0;
    }

    while (true) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= n_B &&
            (Qy == n_A || Qx == 0 || myA[Qy] > myB[Qx - 1])) {

            if (Qx == n_B || Qy == 0 || myA[Qy - 1] <= myB[Qx]) {

                if (Qy < n_A && (Qx == n_B || myA[Qy] <= myB[Qx])) {
                    myM[i] = myA[Qy];
                } else {
                    myM[i] = myB[Qx];
                }
                return;

            } else {
                Kx = Qx + 1;
                Ky = Qy - 1;
            }

        }   else {
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
}

// ----------------------------------------------------------------
// Utilitaire : lance le kernel et retourne le temps d'exécution (ms)
// ----------------------------------------------------------------
// float bench(int* d_A, int* d_B, int* d_M, int sA, int sB, int d, int N) {
//     int threadsPerBlock = (1024 / d) * d;
//     int groupesParBloc  = threadsPerBlock / d;
//     int numBlocks       = (N + groupesParBloc - 1) / groupesParBloc;

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);
//     mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_M, sA, sB, d, N);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float ms = 0;
//     cudaEventElapsedTime(&ms, start, stop);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     return ms;
// }

void verify_result(int* M, int d, int N){
    for (int batch = 0; batch < N ; batch++){
        for (int i = 0; i < d-1; i++){
            assert(M[i] < M[i+1]);
        }
    }
}

int main() {

    // // ---- TEST DE CORRECTION ----------------------------------------
    // printf("=== TEST DE CORRECTION (N=4, d=8) ===\n");
    // {
    //     int N = 4, sA = 4, sB = 4, d = sA + sB;

    //     int h_A[] = { 1,  5,  9, 13,
    //                   2,  6, 10, 14,
    //                   3,  7, 11, 15,
    //                   4,  8, 12, 16 };
    //     int h_B[] = { 0,  3,  7, 11,
    //                   1,  4,  8, 12,
    //                   2,  5,  9, 13,
    //                   3,  6, 10, 14 };
    //     int *h_M = (int*)malloc(N * d * sizeof(int));

    //     int *d_A, *d_B, *d_M;
    //     cudaMalloc(&d_A, N * sA * sizeof(int));
    //     cudaMalloc(&d_B, N * sB * sizeof(int));
    //     cudaMalloc(&d_M, N * d  * sizeof(int));
    //     cudaMemcpy(d_A, h_A, N * sA * sizeof(int), cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_B, h_B, N * sB * sizeof(int), cudaMemcpyHostToDevice);

    //     bench(d_A, d_B, d_M, sA, sB, d, N);
    //     cudaMemcpy(h_M, d_M, N * d * sizeof(int), cudaMemcpyDeviceToHost);

    //     for (int p = 0; p < N; p++) {
    //         printf("M%d = ", p + 1);
    //         for (int j = 0; j < d; j++) printf("%d ", h_M[p * d + j]);
    //         printf("\n");
    //     }
    //     // Attendu :
    //     // M1 = 0 1 3 5 7 9 11 13
    //     // M2 = 1 2 4 6 8 10 12 14
    //     // M3 = 2 3 5 7 9 11 13 15
    //     // M4 = 3 4 6 8 10 12 14 16

    //     cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
    //     free(h_M);
    // }

    Timer Tim;
    int n_A, n_B, n_batch, d;

    printf("\n========== Hand made test ==========\n");
    printf("Number of batchs : ");
    std::cin >> n_batch;
    printf("Number of elements in each array A (<1024): ");
    std::cin >> n_A;
    printf("Number of elements in each array B (<1024 - n_A): ");
    std::cin >> n_B;

    d = n_A + n_B;
    assert(d <= 1024);
    
    int* h_A = new int[n_batch * n_A];
    int* h_B = new int[n_batch * n_B];
    int* h_M = new int[n_batch * d];

    // Generate random arrays
    generate(h_A, h_A + (n_batch * n_A), []() { return rand() % 1000; });
    generate(h_B, h_B + (n_batch * n_B), []() { return rand() % 1000; });

    // 2. Sort the arrays per group before merging them
    for (int batch = 0; batch < n_batch; batch++){
        std::sort(h_A + (batch * n_A), h_A + (batch + 1) * n_A );
        std::sort(h_B + (batch * n_B), h_B + (batch + 1) * n_B );
    }
    
    // CPU sort/merge
    Tim.start();
    mergeSmallBatch_CPU(h_A, h_B, d, n_batch, h_M);
    Tim.add();
    float cpu_time_merge = Tim.getsum();
    verify_result(h_M, d, n_batch);
    printf("CPU Merge completed in %f s.\n", cpu_time_merge);

    // GPU sort/merge
    int *d_A, *d_B, *d_M;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, (n_batch * n_A) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, (n_batch * n_B) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, (n_batch * d) * sizeof(int)));
    
    // Copying arrays from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (n_batch * n_A) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (n_batch * n_B) * sizeof(int), cudaMemcpyHostToDevice));
    
    Tim.reset(); Tim.start();
    
    int threadsPerBlock = (1024 / d) * d;
    int groupesPerBlock  = threadsPerBlock / d;
    int numBlocks       = (n_batch + groupesPerBlock - 1) / groupesPerBlock;
    mergeSmallBatch_k <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_M, n_A, n_B, d, n_batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    Tim.add();
    float gpu_time_merge = Tim.getsum();

    // Copy back to the host
    CUDA_CHECK(cudaMemcpy(h_M, d_M, (n_batch * d) * sizeof(int), cudaMemcpyDeviceToHost));
    Tim.add();
    float gpu_time_merge_to_cpu = Tim.getsum();

    // Check result
    verify_result(h_M, d, n_batch);

    // Printing results
    printf("\nGPU Merge completed in %f s.\n", gpu_time_merge);
    printf("GPU Merge + transfer to CPU completed in %f s.\n", gpu_time_merge_to_cpu);

    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_M));

    // FIXED: Prevent memory leaks
    delete[] h_A;
    delete[] h_B;
    delete[] h_M;

    // ---- ÉTUDE DU TEMPS EN FONCTION DE d --------------------------
    // On fixe le nombre total d'éléments dans M (totalElems = 2^24)
    // et on fait varier d. N = totalElems / d varie donc en sens inverse.
    // Cela garde la quantité de travail et la mémoire constantes (~32 Mo),
    // ce qui rend la comparaison équitable entre les valeurs de d.
    // Pour chaque d, on suppose sA = sB = d/2.
    // On mesure le temps kernel avec cudaEvents (exclut les transferts).
    // printf("\n=== TEMPS EN FONCTION DE d (totalElems=2^24) ===\n");
    // printf("%-8s %-12s %-10s %-14s %-10s\n", "d", "N", "tpb", "groupes/bloc", "temps(ms)");

    // int ds[]      = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    // int nd        = sizeof(ds) / sizeof(ds[0]);
    // int totalElems = 1 << 24;  // nombre total d'éléments dans M (fixe)

    // for (int k = 0; k < nd; k++) {
    //     int d  = ds[k];
    //     int sA = d / 2;
    //     int sB = d / 2;
    //     int N  = totalElems / d;  // nombre de fusions (diminue quand d augmente)

    //     int *d_A, *d_B, *d_M;
    //     cudaMalloc(&d_A, (long)N * sA * sizeof(int));
    //     cudaMalloc(&d_B, (long)N * sB * sizeof(int));
    //     cudaMalloc(&d_M, (long)N * d  * sizeof(int));

    //     cudaMemset(d_A, 0, (long)N * sA * sizeof(int));
    //     cudaMemset(d_B, 0, (long)N * sB * sizeof(int));

    //     int tpb            = (1024 / d) * d;
    //     int groupesParBloc = tpb / d;

    //     float ms = bench(d_A, d_B, d_M, sA, sB, d, N);
    //     printf("%-8d %-12d %-10d %-14d %.3f\n", d, N, tpb, groupesParBloc, ms);

    //     cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
    // }

    return 0;
}