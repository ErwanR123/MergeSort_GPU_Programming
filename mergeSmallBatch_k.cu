#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ================================================================
// KERNEL : mergeSmallBatch_k
// Fusionne en batch N paires (Ai, Bi) → Mi, chaque paire ayant
// une taille totale d = |Ai| + |Bi| ≤ 1024.
//
// Principe : on réutilise mergeSmall_k mais chaque "groupe" de
// d threads consécutifs s'occupe d'une paire (Ai, Bi).
// Plusieurs groupes peuvent cohabiter dans un même bloc CUDA.
//
// Lancement :
//   threadsPerBlock = multiple de d, ≤ 1024  (ex: (1024/d)*d)
//   numBlocks = ceil(N / (threadsPerBlock/d))
//   mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(...)
//
// Mémoire : tous les Ai sont concaténés dans un seul tableau A
//           (idem pour Bi et Mi).
//   A = [A1 | A2 | ... | AN],  chaque Ai a une taille sA = d/2
//   B = [B1 | B2 | ... | BN],  chaque Bi a une taille sB = d/2
//   M = [M1 | M2 | ... | MN],  chaque Mi a une taille d
//
// Pour simplifier, on suppose ici |Ai| = |Bi| = d/2 pour tout i.
// ================================================================
__global__ void mergeSmallBatch_k(int* A, int* B, int* M,
                                   int sA, int sB, int d, int N) {

    // ---- BLOC 1 : indices de virtualisation --------------------
    // Qt  = numéro du "groupe" LOCAL dans ce bloc
    //        (chaque groupe = d threads qui fusionnent une paire)
    // tidx = index LOCAL de ce thread dans son groupe (0..d-1)
    // gbx  = numéro du groupe GLOBAL (= quel couple Ai,Bi je traite)
    //
    // Exemple : threadsPerBlock=512, d=128
    //   bloc 0 : Qt=0→3, gbx=0→3   (4 groupes dans le bloc 0)
    //   bloc 1 : Qt=0→3, gbx=4→7   (4 groupes dans le bloc 1)
    int Qt   = threadIdx.x / d;                      // groupe local dans le bloc
    int tidx = threadIdx.x - (Qt * d);               // = threadIdx.x % d
    int gbx  = Qt + blockIdx.x * (blockDim.x / d);   // groupe global

    // Sécurité : si le groupe global dépasse N, ce thread n'a rien à faire
    if (gbx >= N) return;

    // ---- BLOC 2 : pointer vers le bon couple (Agbx, Bgbx) -----
    // Chaque groupe gbx travaille sur :
    //   A[gbx*sA .. gbx*sA + sA - 1]  →  le tableau Agbx
    //   B[gbx*sB .. gbx*sB + sB - 1]  →  le tableau Bgbx
    //   M[gbx*d  .. gbx*d  + d  - 1]  →  le résultat Mgbx
    int* myA = A + gbx * sA;   // début de Ai dans le grand tableau A
    int* myB = B + gbx * sB;   // début de Bi dans le grand tableau B
    int* myM = M + gbx * d;    // début de Mi dans le grand tableau M

    // ---- BLOC 3 : merge path identique à mergeSmall_k ---------
    // tidx = position de ce thread dans M_gbx (sa diagonale locale)
    int i = tidx;
    if (i >= d) return;  // sécurité (ne devrait pas arriver)

    // Bornes de la diagonale (identique à mergeSmall_k)
    int Kx, Ky, Px, Py;

    if (i > sA) {
        Kx = i - sA;   Ky = sA;
        Px = sA;        Py = i - sA;
    } else {
        Kx = 0;   Ky = i;
        Px = i;   Py = 0;
    }

    // Recherche binaire sur la diagonale
    while (1) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= sB &&
            (Qy == sA || Qx == 0 || myA[Qy] > myB[Qx - 1])) {

            if (Qx == sB || Qy == 0 || myA[Qy - 1] <= myB[Qx]) {

                if (Qy < sA && (Qx == sB || myA[Qy] <= myB[Qx])) {
                    myM[i] = myA[Qy];
                } else {
                    myM[i] = myB[Qx];
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

// ================================================================
// MAIN — test de mergeSmallBatch_k
// ================================================================
int main() {

    // ---- Paramètres --------------------------------------------
    int N  = 4;       // nombre de paires à fusionner
    int sA = 4;       // taille de chaque Ai
    int sB = 4;       // taille de chaque Bi
    int d  = sA + sB; // = 8 éléments par paire fusionnée

    // ---- Données host ------------------------------------------
    // A = [A1 | A2 | A3 | A4], chaque Ai trié, taille sA=4
    int h_A[] = { 1, 5, 9, 13,     // A1
                  2, 6, 10, 14,    // A2
                  3, 7, 11, 15,    // A3
                  4, 8, 12, 16 };  // A4

    // B = [B1 | B2 | B3 | B4], chaque Bi trié, taille sB=4
    int h_B[] = { 0, 3, 7, 11,     // B1
                  1, 4, 8, 12,     // B2
                  2, 5, 9, 13,     // B3
                  3, 6, 10, 14 };  // B4

    int h_M[N * d];  // résultat

    // ---- Allocation et copie GPU --------------------------------
    int *d_A, *d_B, *d_M;
    cudaMalloc(&d_A, N * sA * sizeof(int));
    cudaMalloc(&d_B, N * sB * sizeof(int));
    cudaMalloc(&d_M, N * d  * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sB * sizeof(int), cudaMemcpyHostToDevice);

    // ---- Lancement du kernel -----------------------------------
    // threadsPerBlock = plus grand multiple de d ≤ 1024
    // Ici d=8, donc threadsPerBlock = 1024 (1024/8 = 128 groupes par bloc)
    int threadsPerBlock = (1024 / d) * d;  // = 1024 ici
    int groupesParBloc  = threadsPerBlock / d;
    int numBlocks = (N + groupesParBloc - 1) / groupesParBloc;  // = ceil(N/128) = 1

    printf("d=%d, threadsPerBlock=%d, groupesParBloc=%d, numBlocks=%d\n",
           d, threadsPerBlock, groupesParBloc, numBlocks);

    mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(
        d_A, d_B, d_M, sA, sB, d, N);

    // ---- Récupération et affichage ------------------------------
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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);

    return 0;
}