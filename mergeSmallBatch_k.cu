#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Fusionne en batch N paires (Ai, Bi) → Mi, chaque paire ayant une taille
// totale d = |Ai| + |Bi| <= 1024. Tous les Ai sont concaténés dans A (idem B, M).
//
// Lancement : mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(...)
//   threadsPerBlock = multiple de d, <= 1024
//   numBlocks       = ceil(N / (threadsPerBlock / d))
//
// Importance des trois indices :
//
//   Qt = threadIdx.x / d
//     Identifie le groupe LOCAL auquel appartient ce thread dans son bloc.
//     Chaque groupe de d threads consécutifs est responsable d'une fusion.
//     Sans Qt, un thread ne saurait pas à quelle fusion il appartient dans le bloc.
//
//   tidx = threadIdx.x - Qt*d
//     C'est l'équivalent de i = threadIdx.x dans mergeSmall_k : le numéro de
//     diagonale LOCAL à cette fusion. Remis à zéro à chaque nouvelle fusion.
//     Sans tidx, le thread 5 d'un groupe croirait être sur la diagonale 5
//     de M global, alors qu'il est sur la diagonale 1 de sa fusion locale.
//
//   gbx = Qt + blockIdx.x * (blockDim.x / d)
//     Numéro GLOBAL de la fusion parmi les N paires. Permet de calculer
//     les offsets myA, myB, myM pour pointer vers la bonne paire (Ai, Bi).
//     Sans gbx, tous les groupes travailleraient sur la même paire.
//
// Ces trois indices permettent de "virtualiser" le kernel mergeSmall_k :
// chaque thread se comporte comme s'il était dans un kernel mono-fusion,
// mais en réalité il partage son bloc avec d'autres fusions.
__global__ void mergeSmallBatch_k(int* A, int* B, int* M,
                                   int sA, int sB, int d, int N) {

    int Qt   = threadIdx.x / d;
    int tidx = threadIdx.x - (Qt * d);
    int gbx  = Qt + blockIdx.x * (blockDim.x / d);

    if (gbx >= N) return;

    // Pointeurs vers la paire gbx dans les tableaux batch
    int* myA = A + gbx * sA;
    int* myB = B + gbx * sB;
    int* myM = M + gbx * d;

    // Merge path identique à mergeSmall_k, avec tidx comme numéro de diagonale
    int i = tidx;

    int Kx, Ky, Px, Py;
    if (i > sA) {
        Kx = i - sA;  Ky = sA;
        Px = sA;      Py = i - sA;
    } else {
        Kx = 0;  Ky = i;
        Px = i;  Py = 0;
    }

    while (1) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= sB &&
            (Qy == sA || Qx == 0 || myA[Qy] > myB[Qx - 1])) {

            if (Qx == sB || Qy == 0 || myA[Qy - 1] <= myB[Qx]) {
                myM[i] = (Qy < sA && (Qx == sB || myA[Qy] <= myB[Qx]))
                         ? myA[Qy] : myB[Qx];
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

// ----------------------------------------------------------------
// Utilitaire : lance le kernel et retourne le temps d'exécution (ms)
// ----------------------------------------------------------------
float bench(int* d_A, int* d_B, int* d_M, int sA, int sB, int d, int N) {
    int threadsPerBlock = (1024 / d) * d;
    int groupesParBloc  = threadsPerBlock / d;
    int numBlocks       = (N + groupesParBloc - 1) / groupesParBloc;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_M, sA, sB, d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {

    // ---- TEST DE CORRECTION ----------------------------------------
    printf("=== TEST DE CORRECTION (N=4, d=8) ===\n");
    {
        int N = 4, sA = 4, sB = 4, d = sA + sB;

        int h_A[] = { 1,  5,  9, 13,
                      2,  6, 10, 14,
                      3,  7, 11, 15,
                      4,  8, 12, 16 };
        int h_B[] = { 0,  3,  7, 11,
                      1,  4,  8, 12,
                      2,  5,  9, 13,
                      3,  6, 10, 14 };
        int *h_M = (int*)malloc(N * d * sizeof(int));

        int *d_A, *d_B, *d_M;
        cudaMalloc(&d_A, N * sA * sizeof(int));
        cudaMalloc(&d_B, N * sB * sizeof(int));
        cudaMalloc(&d_M, N * d  * sizeof(int));
        cudaMemcpy(d_A, h_A, N * sA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * sB * sizeof(int), cudaMemcpyHostToDevice);

        bench(d_A, d_B, d_M, sA, sB, d, N);
        cudaMemcpy(h_M, d_M, N * d * sizeof(int), cudaMemcpyDeviceToHost);

        for (int p = 0; p < N; p++) {
            printf("M%d = ", p + 1);
            for (int j = 0; j < d; j++) printf("%d ", h_M[p * d + j]);
            printf("\n");
        }
        // Attendu :
        // M1 = 0 1 3 5 7 9 11 13
        // M2 = 1 2 4 6 8 10 12 14
        // M3 = 2 3 5 7 9 11 13 15
        // M4 = 3 4 6 8 10 12 14 16

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
        free(h_M);
    }

    // ---- ÉTUDE DU TEMPS EN FONCTION DE d --------------------------
    // On fixe le nombre total d'éléments dans M (totalElems = 2^24)
    // et on fait varier d. N = totalElems / d varie donc en sens inverse.
    // Cela garde la quantité de travail et la mémoire constantes (~32 Mo),
    // ce qui rend la comparaison équitable entre les valeurs de d.
    // Pour chaque d, on suppose sA = sB = d/2.
    // On mesure le temps kernel avec cudaEvents (exclut les transferts).
    printf("\n=== TEMPS EN FONCTION DE d (totalElems=2^24) ===\n");
    printf("%-8s %-12s %-10s %-14s %-10s\n", "d", "N", "tpb", "groupes/bloc", "temps(ms)");

    int ds[]      = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int nd        = sizeof(ds) / sizeof(ds[0]);
    int totalElems = 1 << 24;  // nombre total d'éléments dans M (fixe)

    for (int k = 0; k < nd; k++) {
        int d  = ds[k];
        int sA = d / 2;
        int sB = d / 2;
        int N  = totalElems / d;  // nombre de fusions (diminue quand d augmente)

        int *d_A, *d_B, *d_M;
        cudaMalloc(&d_A, (long)N * sA * sizeof(int));
        cudaMalloc(&d_B, (long)N * sB * sizeof(int));
        cudaMalloc(&d_M, (long)N * d  * sizeof(int));

        cudaMemset(d_A, 0, (long)N * sA * sizeof(int));
        cudaMemset(d_B, 0, (long)N * sB * sizeof(int));

        int tpb            = (1024 / d) * d;
        int groupesParBloc = tpb / d;

        float ms = bench(d_A, d_B, d_M, sA, sB, d, N);
        printf("%-8d %-12d %-10d %-14d %.3f\n", d, N, tpb, groupesParBloc, ms);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
    }

    return 0;
}
