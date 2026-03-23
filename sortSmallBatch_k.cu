#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ================================================================
// KERNEL UTILITAIRE : mergeSmallBatch_k (identique à 2.a)
// Fusionne en batch : pour chaque groupe gbx, fusionne
// A[gbx*sA .. ] et B[gbx*sB .. ] dans M[gbx*d .. ]
// ================================================================
__global__ void mergeSmallBatch_k(int* A, int* B, int* M,
                                   int sA, int sB, int d, int N) {

    int Qt   = threadIdx.x / d;
    int tidx = threadIdx.x - (Qt * d);
    int gbx  = Qt + blockIdx.x * (blockDim.x / d);

    if (gbx >= N) return;

    int* myA = A + gbx * sA;
    int* myB = B + gbx * sB;
    int* myM = M + gbx * d;

    int i = tidx;
    if (i >= d) return;

    int Kx, Ky, Px, Py;

    if (i > sA) {
        Kx = i - sA;  Ky = sA;
        Px = sA;       Py = i - sA;
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
                if (Qy < sA && (Qx == sB || myA[Qy] <= myB[Qx])) {
                    myM[i] = myA[Qy];
                } else {
                    myM[i] = myB[Qx];
                }
                return;
            } else {
                Kx = Qx + 1;  Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1;  Py = Qy + 1;
        }
    }
}

// ================================================================
// KERNEL : sortSmallBatch_k
// Trie N tableaux {Mi} de taille d ≤ 1024 chacun.
//
// Algorithme : merge sort itératif bottom-up
//   - Étape w=1 : chaque élément est un sous-tableau trié de taille 1
//   - Étape w=2 : on fusionne des paires de 1 → sous-tableaux de 2
//   - Étape w=4 : on fusionne des paires de 2 → sous-tableaux de 4
//   - ...
//   - Étape w=d/2 : on fusionne 2 moitiés → tableau trié de taille d
//
// À chaque étape, on appelle mergeSmallBatch_k pour fusionner
// toutes les paires en parallèle.
//
// On alterne entre deux buffers (ping-pong) pour éviter de
// lire et écrire au même endroit.
// ================================================================
void sortSmallBatch(int* d_data, int* d_temp, int d, int N) {

    // Merge sort bottom-up : on double la taille des sous-tableaux à chaque étape
    // w = taille d'un sous-tableau à fusionner (commence à 1)
    // mergeSize = taille après fusion = 2*w
    for (int w = 1; w < d; w *= 2) {

        int mergeSize = 2 * w;  // taille de chaque fusion

        // Nombre total de fusions à faire :
        //   - chaque tableau Mi a d éléments
        //   - dans chaque Mi, il y a d / mergeSize fusions
        //   - il y a N tableaux
        // Donc N_merges = N * (d / mergeSize)
        int fusionsParTableau = d / mergeSize;
        int N_merges = N * fusionsParTableau;

        // Pour mergeSmallBatch_k :
        //   sA = sB = w  (taille de chaque moitié)
        //   d_merge = mergeSize = 2*w
        //
        // Les "Ai" et "Bi" sont des sous-tableaux CONTIGUS dans d_data :
        //   Pour le merge j dans le tableau i :
        //     Ai = d_data[i*d + j*mergeSize ..  i*d + j*mergeSize + w - 1]
        //     Bi = d_data[i*d + j*mergeSize + w .. i*d + j*mergeSize + 2w - 1]
        //
        // Astuce : on voit d_data comme N_merges paires consécutives
        // de taille w chacune. On passe l'adresse de base et
        // mergeSmallBatch_k s'en occupe grâce à gbx.

        int threadsPerBlock = (1024 / mergeSize) * mergeSize;
        if (threadsPerBlock == 0) threadsPerBlock = mergeSize;  // si mergeSize > 1024
        int groupesParBloc = threadsPerBlock / mergeSize;
        int numBlocks = (N_merges + groupesParBloc - 1) / groupesParBloc;

        // d_data contient les sous-tableaux entrelacés :
        // [A0|B0|A1|B1|...] avec Ai de taille w et Bi de taille w
        // On passe A = d_data (décalage 0) et B = d_data + w (décalage w)
        // Mais mergeSmallBatch_k attend A et B séparés avec un stride sA et sB.
        //
        // Solution : on utilise un kernel adapté qui lit les paires
        // directement depuis d_data avec le bon stride.

        // Version simplifiée : on utilise un seul tableau source,
        // chaque "paire" est [A|B] contigu de taille mergeSize.
        // A_i commence à source + i*mergeSize, B_i à source + i*mergeSize + w.

        mergeSmallBatch_k<<<numBlocks, threadsPerBlock>>>(
            d_data,          // A : base des moitiés gauches
            d_data + w,      // B : base des moitiés droites (décalé de w)
            d_temp,          // M : résultat dans le buffer temporaire
            w, w,            // sA = sB = w
            mergeSize,       // d = taille de chaque fusion
            N_merges);

        // Ping-pong : échanger les buffers pour l'étape suivante
        int* swap = d_data;
        d_data = d_temp;
        d_temp = swap;
    }

    // ATTENTION : après la boucle, le résultat est dans d_data
    // (qui a peut-être été swappé). L'appelant doit en tenir compte.
}

// ================================================================
// MAIN — test de sortSmallBatch
// ================================================================
int main() {

    int N = 4;   // nombre de tableaux à trier
    int d = 8;   // taille de chaque tableau

    // Données non triées : N=4 tableaux de d=8 éléments
    int h_data[] = { 13, 1, 9, 5, 11, 3, 7, 0,    // M1 (non trié)
                     14, 2, 10, 6, 12, 4, 8, 1,   // M2
                     15, 3, 11, 7, 13, 5, 9, 2,   // M3
                     16, 4, 12, 8, 14, 6, 10, 3 }; // M4

    int totalSize = N * d;
    int h_result[32];

    // ---- GPU ---------------------------------------------------
    int *d_data, *d_temp;
    cudaMalloc(&d_data, totalSize * sizeof(int));
    cudaMalloc(&d_temp, totalSize * sizeof(int));

    cudaMemcpy(d_data, h_data, totalSize * sizeof(int), cudaMemcpyHostToDevice);

    // ---- Tri ---------------------------------------------------
    // sortSmallBatch modifie d_data et d_temp par ping-pong.
    // On doit savoir dans quel buffer est le résultat final.
    // Nombre d'étapes = log2(d) = 3 (pour d=8 : w=1,2,4)
    // Si log2(d) est impair, le résultat est dans d_temp (l'original d_data a été swappé)
    // Si log2(d) est pair, le résultat est dans d_data

    // On sauvegarde le pointeur original pour savoir où est le résultat
    int* original_data = d_data;
    int* original_temp = d_temp;

    sortSmallBatch(d_data, d_temp, d, N);

    // Compter le nombre d'étapes pour savoir quel buffer contient le résultat
    int steps = 0;
    for (int w = 1; w < d; w *= 2) steps++;

    int* result_ptr = (steps % 2 == 0) ? original_data : original_temp;

    cudaMemcpy(h_result, result_ptr, totalSize * sizeof(int), cudaMemcpyDeviceToHost);

    // ---- Affichage ---------------------------------------------
    for (int p = 0; p < N; p++) {
        printf("M%d trié = ", p + 1);
        for (int j = 0; j < d; j++) printf("%d ", h_result[p * d + j]);
        printf("\n");
    }
    // Attendu :
    // M1 trié = 0 1 3 5 7 9 11 13
    // M2 trié = 1 2 4 6 8 10 12 14
    // M3 trié = 2 3 5 7 9 11 13 15
    // M4 trié = 3 4 6 8 10 12 14 16

    cudaFree(original_data);
    cudaFree(original_temp);

    return 0;
}