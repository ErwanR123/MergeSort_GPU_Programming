#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================================
// Approche naive (sous-optimale) : extension directe de mergeSmall_k
// On remplace threadIdx.x par idx = threadIdx.x + blockIdx.x * blockDim.x.
// Chaque thread calcule M[idx] par une recherche binaire sur la TOTALITE
// de A et B en memoire globale.
//
// Pourquoi c'est sous-optimal :
//   1. Recherche binaire en O(log(sA+sB)) au lieu de O(log T).
//   2. Acces a la memoire globale avec des sauts irreguliers (pas de
//      coalescence, pas de reutilisation via shared memory).
//   3. Les threads d'un meme bloc accdent a des zones tres eloignees de
//      A et B => pas de benefice du cache L1/shared memory.
// ============================================================
__global__ void mergeLarge_naive(int* A, int sA, int* B, int sB, int* M) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sA + sB) return;

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
                if (Qy < sA && (Qx == sB || A[Qy] <= B[Qx])) {
                    M[i] = A[Qy];
                } else {
                    M[i] = B[Qx];
                }
                return;
            } else {
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
}

// Kernel 1 - Partitioning
// Thread b trouve l'intersection de la diagonale (b*T) avec le merge path.
// Resultat stocke dans Adiag[b] et Bdiag[b].
__global__ void mergeLarge_partition(int* A, int sA, int* B, int sB,
                                     int* Adiag, int* Bdiag, int T) {
    int b    = threadIdx.x + blockIdx.x * blockDim.x;
    int diag = b * T;
    if (diag > sA + sB) return;

    int Kx, Ky, Px, Py;
    if (diag > sA) {
        Kx = diag - sA; Ky = sA;
        Px = sA;        Py = diag - sA;
    } else {
        Kx = 0;    Ky = diag;
        Px = diag; Py = 0;
    }

    while (1) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= sB &&
            (Qy == sA || Qx == 0 || A[Qy] > B[Qx - 1])) {
            if (Qx == sB || Qy == 0 || A[Qy - 1] <= B[Qx]) {
                Adiag[b] = Qy;
                Bdiag[b] = Qx;
                return;
            } else {
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
}

// Kernel 2 - Merging
// Le bloc b fusionne A[Adiag[b]..Adiag[b+1]-1] et B[Bdiag[b]..Bdiag[b+1]-1]
// en chargeant d'abord sa tranche en shared memory, puis chaque thread
// fait sa binary search localement.
__global__ void mergeLarge_merge(int* A, int* B, int* M,
                                 int* Adiag, int* Bdiag, int T) {
    extern __shared__ int smem[];  // 2*T ints : T pour A, T pour B

    int b    = blockIdx.x;
    int tidx = threadIdx.x;

    int sA = Adiag[b + 1] - Adiag[b];
    int sB = Bdiag[b + 1] - Bdiag[b];

    int* shA = smem;
    int* shB = smem + T;

    // Chargement cooperatif depuis la memoire globale (acces coalescents)
    if (tidx < sA) shA[tidx] = A[Adiag[b] + tidx];
    if (tidx < sB) shB[tidx] = B[Bdiag[b] + tidx];
    __syncthreads();

    // Binary search en shared memory, identique a mergeSmall_k
    int i = tidx;
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
            (Qy == sA || Qx == 0 || shA[Qy] > shB[Qx - 1])) {
            if (Qx == sB || Qy == 0 || shA[Qy - 1] <= shB[Qx]) {
                if (Qy < sA && (Qx == sB || shA[Qy] <= shB[Qx])) {
                    M[b * T + i] = shA[Qy];
                } else {
                    M[b * T + i] = shB[Qx];
                }
                return;
            } else {
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
}

int main() {

    // ---- TEST DE CORRECTION ----------------------------------------
    printf("=== TEST DE CORRECTION ===\n");
    {
        int A[] = {1, 5, 6, 9, 12, 15, 18, 20};
        int B[] = {2, 4, 7, 10, 11, 13, 16, 19};
        int sA = 8, sB = 8, n = sA + sB;
        int T  = 4;  // threads par bloc => 4 blocs
        int P  = n / T;

        int *d_A, *d_B, *d_M, *d_Adiag, *d_Bdiag;
        cudaMalloc(&d_A,     sA * sizeof(int));
        cudaMalloc(&d_B,     sB * sizeof(int));
        cudaMalloc(&d_M,     n  * sizeof(int));
        cudaMalloc(&d_Adiag, (P + 1) * sizeof(int));
        cudaMalloc(&d_Bdiag, (P + 1) * sizeof(int));
        cudaMemcpy(d_A, A, sA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sB * sizeof(int), cudaMemcpyHostToDevice);

        mergeLarge_partition<<<1, P + 1>>>(d_A, sA, d_B, sB, d_Adiag, d_Bdiag, T);
        mergeLarge_merge<<<P, T, 2 * T * sizeof(int)>>>(d_A, d_B, d_M, d_Adiag, d_Bdiag, T);

        int M[16];
        cudaMemcpy(M, d_M, n * sizeof(int), cudaMemcpyDeviceToHost);
        printf("M   = ");
        for (int i = 0; i < n; i++) printf("%d ", M[i]);
        printf("\nref = 1 2 4 5 6 7 9 10 11 12 13 15 16 18 19 20\n");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
        cudaFree(d_Adiag); cudaFree(d_Bdiag);
    }

    // sA = sB = d/2, d varie de 2^20 a 2^27, T = 512 threads par bloc
    int T = 512;
    int sizes[] = {1<<20, 1<<21, 1<<22, 1<<23, 1<<24, 1<<25, 1<<26, 1<<27};
    int ns = sizeof(sizes) / sizeof(sizes[0]);

    // Tableaux pour stocker les temps et les afficher en deux tableaux separes
    float ms_naive[8], ms_part[8], ms_merge[8];

    for (int k = 0; k < ns; k++) {
        int n  = sizes[k];
        int sA = n / 2, sB = n / 2;
        int P  = n / T;

        int *d_A, *d_B, *d_M, *d_Adiag, *d_Bdiag;
        cudaMalloc(&d_A,     (long)sA * sizeof(int));
        cudaMalloc(&d_B,     (long)sB * sizeof(int));
        cudaMalloc(&d_M,     (long)n  * sizeof(int));
        cudaMalloc(&d_Adiag, (P + 1)  * sizeof(int));
        cudaMalloc(&d_Bdiag, (P + 1)  * sizeof(int));
        cudaMemset(d_A, 0, (long)sA * sizeof(int));
        cudaMemset(d_B, 0, (long)sB * sizeof(int));

        cudaEvent_t s0, e0, s1, e1, s2, e2;
        cudaEventCreate(&s0); cudaEventCreate(&e0);
        cudaEventCreate(&s1); cudaEventCreate(&e1);
        cudaEventCreate(&s2); cudaEventCreate(&e2);

        int threadsNaive = 256;
        int blocksNaive  = (n + threadsNaive - 1) / threadsNaive;
        cudaEventRecord(s0);
        mergeLarge_naive<<<blocksNaive, threadsNaive>>>(d_A, sA, d_B, sB, d_M);
        cudaEventRecord(e0);
        cudaEventSynchronize(e0);

        cudaEventRecord(s1);
        mergeLarge_partition<<<(P + 1 + 255) / 256, 256>>>(d_A, sA, d_B, sB, d_Adiag, d_Bdiag, T);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);

        cudaEventRecord(s2);
        mergeLarge_merge<<<P, T, 2 * T * sizeof(int)>>>(d_A, d_B, d_M, d_Adiag, d_Bdiag, T);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);

        cudaEventElapsedTime(&ms_naive[k], s0, e0);
        cudaEventElapsedTime(&ms_part[k],  s1, e1);
        cudaEventElapsedTime(&ms_merge[k], s2, e2);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
        cudaFree(d_Adiag); cudaFree(d_Bdiag);
        cudaEventDestroy(s0); cudaEventDestroy(e0);
        cudaEventDestroy(s1); cudaEventDestroy(e1);
        cudaEventDestroy(s2); cudaEventDestroy(e2);
    }

    // --- Tableau 1 : Naive vs 2-etapes ---
    printf("\n=== Tableau 1 : Naive vs 2-etapes (T=512) ===\n");
    printf("%-12s %-14s %-14s\n", "d", "naive(ms)", "2-etapes(ms)");
    for (int k = 0; k < ns; k++)
        printf("%-12d %-14.3f %-14.3f\n",
               sizes[k], ms_naive[k], ms_part[k] + ms_merge[k]);

    // --- Tableau 2 : Decomposition des 2 etapes ---
    printf("\n=== Tableau 2 : Decomposition 2-etapes (T=512) ===\n");
    printf("%-12s %-16s %-14s\n", "d", "partition(ms)", "merge(ms)");
    for (int k = 0; k < ns; k++)
        printf("%-12d %-16.3f %-14.3f\n",
               sizes[k], ms_part[k], ms_merge[k]);

    return 0;
}
