#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// ================================================================
// QUESTION 3 : Merge de grands tableaux avec partitionnement
//
// Stratégie en 2 étapes (cf. [6] Green et al. 2012) :
//
//   ÉTAPE 1 — Partitionnement (1 kernel) :
//     Chaque bloc CUDA va traiter un morceau de taille CHUNK
//     du tableau de sortie M. Pour savoir QUEL morceau de A
//     et QUEL morceau de B lui correspondent, on fait une
//     recherche binaire sur la diagonale correspondante.
//     → On stocke les points de découpe dans d_Adiag[] et d_Bdiag[].
//
//   ÉTAPE 2 — Merge local (1 kernel) :
//     Chaque bloc lit son morceau de A et son morceau de B
//     en mémoire partagée (shared memory), puis fait le merge
//     path localement avec ses threads pour remplir son morceau
//     de M.
//
// Avantage par rapport à la version "naïve" (idx global) :
//   - Les lectures mémoire sont coalescées (shared memory)
//   - Le travail est parfaitement réparti entre les blocs
//   - On utilise la mémoire partagée pour les accès aléatoires
// ================================================================

// Taille du morceau traité par chaque bloc dans l'étape de merge
// = nombre de threads par bloc (chaque thread produit 1 élément de M)
#define CHUNK 512

// ================================================================
// KERNEL 1 : partitionKernel
// Pour chaque bloc k, on cherche le point d'intersection du
// merge path avec la diagonale k*CHUNK.
// Résultat : d_Adiag[k] = combien d'éléments de A sont AVANT
//            cette diagonale, idem d_Bdiag[k] pour B.
//
// Nombre de blocs nécessaires : numParts = ceil((sA+sB) / CHUNK) + 1
// On cherche numParts+1 points de partition (y compris les bornes).
// ================================================================
__global__ void partitionKernel(int* A, int sA,
                                 int* B, int sB,
                                 int* d_Adiag, int* d_Bdiag,
                                 int numParts) {

    // Chaque thread calcule UN point de partition
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k > numParts) return;  // numParts+1 points au total (de 0 à numParts)

    // La diagonale k correspond à la position k*CHUNK dans M
    // (sauf la dernière qui correspond à sA+sB)
    int diag = min((long long)k * CHUNK, (long long)(sA + sB));

    // ---- Recherche binaire sur la diagonale --------------------
    // On cherche combien d'éléments de A (= ai) et de B (= bi)
    // sont situés AVANT la position diag dans M.
    // Contrainte : ai + bi = diag
    //
    // Bornes de la recherche : ai ∈ [atop_min, atop_max]
    //   atop_max = min(diag, sA)   → on ne peut pas prendre plus que sA éléments de A
    //   atop_min = diag - sB       → si diag > sB, on doit prendre au moins diag-sB de A
    //                                 (clamped à 0)
    int atop = (diag > sA) ? sA : diag;   // borne haute de ai
    int abot = (diag > sB) ? (diag - sB) : 0;  // borne basse de ai

    while (atop > abot) {
        int amid = (atop + abot) / 2;
        int bmid = diag - amid;

        // On teste si le point (amid, bmid) est sur le merge path
        // Condition : A[amid] > B[bmid-1] signifie qu'on est au-dessus ou sur le chemin
        if (A[amid] > B[bmid - 1]) {
            // Le chemin est plus haut → on descend atop
            atop = amid;
        } else {
            // Le chemin est plus bas → on monte abot
            abot = amid + 1;
        }
    }

    // atop = abot = ai = nombre d'éléments de A avant cette diagonale
    d_Adiag[k] = atop;
    d_Bdiag[k] = diag - atop;
}

// ================================================================
// KERNEL 2 : mergeLocalKernel
// Chaque bloc k traite le morceau [k*CHUNK, (k+1)*CHUNK) de M.
// Il sait grâce à d_Adiag et d_Bdiag quels sous-tableaux de A et B
// il doit fusionner. Il les charge en shared memory puis fait
// le merge path localement.
// ================================================================
__global__ void mergeLocalKernel(int* A, int sA,
                                  int* B, int sB,
                                  int* M,
                                  int* d_Adiag, int* d_Bdiag) {

    // ---- BLOC 1 : identifier mon morceau -----------------------
    int k = blockIdx.x;  // numéro du bloc = numéro du morceau

    // Lire les bornes de partition pour ce bloc
    int a_start = d_Adiag[k];      // début dans A pour ce bloc
    int a_end   = d_Adiag[k + 1];  // fin dans A (début du bloc suivant)
    int b_start = d_Bdiag[k];      // début dans B pour ce bloc
    int b_end   = d_Bdiag[k + 1];  // fin dans B

    int localSA = a_end - a_start;  // nombre d'éléments de A pour ce bloc
    int localSB = b_end - b_start;  // nombre d'éléments de B pour ce bloc
    int localN  = localSA + localSB; // taille du morceau de M (≤ CHUNK)

    // ---- BLOC 2 : charger en shared memory ---------------------
    // On copie les sous-tableaux de A et B dans la mémoire partagée
    // pour des accès rapides et aléatoires pendant la recherche binaire.
    __shared__ int shdA[CHUNK];  // au max CHUNK éléments de A
    __shared__ int shdB[CHUNK];  // au max CHUNK éléments de B

    // Chargement coopératif : chaque thread copie un ou plusieurs éléments
    for (int t = threadIdx.x; t < localSA; t += blockDim.x) {
        shdA[t] = A[a_start + t];
    }
    for (int t = threadIdx.x; t < localSB; t += blockDim.x) {
        shdB[t] = B[b_start + t];
    }
    __syncthreads();  // attendre que tous les threads aient fini de charger

    // ---- BLOC 3 : merge path local -----------------------------
    // Identique à mergeSmall_k mais sur shdA/shdB au lieu de A/B
    int i = threadIdx.x;  // position de ce thread dans le morceau local
    if (i >= localN) return;

    int Kx, Ky, Px, Py;

    if (i > localSA) {
        Kx = i - localSA;   Ky = localSA;
        Px = localSA;        Py = i - localSA;
    } else {
        Kx = 0;   Ky = i;
        Px = i;   Py = 0;
    }

    while (1) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= localSB &&
            (Qy == localSA || Qx == 0 || shdA[Qy] > shdB[Qx - 1])) {

            if (Qx == localSB || Qy == 0 || shdA[Qy - 1] <= shdB[Qx]) {

                // Position globale dans M = début du morceau + position locale
                int globalPos = k * CHUNK + i;
                if (globalPos < sA + sB) {  // sécurité
                    if (Qy < localSA && (Qx == localSB || shdA[Qy] <= shdB[Qx])) {
                        M[globalPos] = shdA[Qy];
                    } else {
                        M[globalPos] = shdB[Qx];
                    }
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
// MAIN — test du merge de grands tableaux
// ================================================================
int main() {

    // ---- Paramètres --------------------------------------------
    // On teste avec des tableaux assez grands pour utiliser plusieurs blocs
    int sA = 1 << 20;  // 1M éléments dans A
    int sB = 1 << 20;  // 1M éléments dans B
    int sM = sA + sB;  // 2M éléments dans M

    // ---- Génération des données triées sur le CPU ---------------
    int* h_A = (int*)malloc(sA * sizeof(int));
    int* h_B = (int*)malloc(sB * sizeof(int));
    int* h_M = (int*)malloc(sM * sizeof(int));

    // A = nombres pairs triés : 0, 2, 4, 6, ...
    for (int i = 0; i < sA; i++) h_A[i] = 2 * i;
    // B = nombres impairs triés : 1, 3, 5, 7, ...
    for (int i = 0; i < sB; i++) h_B[i] = 2 * i + 1;

    // ---- Allocation GPU ----------------------------------------
    int *d_A, *d_B, *d_M;
    cudaMalloc(&d_A, sA * sizeof(int));
    cudaMalloc(&d_B, sB * sizeof(int));
    cudaMalloc(&d_M, sM * sizeof(int));

    cudaMemcpy(d_A, h_A, sA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sB * sizeof(int), cudaMemcpyHostToDevice);

    // ---- ÉTAPE 1 : Partitionnement -----------------------------
    // Nombre de morceaux = ceil(sM / CHUNK)
    int numParts = (sM + CHUNK - 1) / CHUNK;

    // Tableaux pour stocker les points de partition
    int *d_Adiag, *d_Bdiag;
    cudaMalloc(&d_Adiag, (numParts + 1) * sizeof(int));
    cudaMalloc(&d_Bdiag, (numParts + 1) * sizeof(int));

    // Lancer le kernel de partitionnement
    // Chaque thread calcule 1 point de partition → numParts+1 threads
    int partThreads = 256;
    int partBlocks = (numParts + 1 + partThreads - 1) / partThreads;

    partitionKernel<<<partBlocks, partThreads>>>(
        d_A, sA, d_B, sB, d_Adiag, d_Bdiag, numParts);

    // ---- ÉTAPE 2 : Merge local par blocs -----------------------
    // Chaque bloc traite un morceau de CHUNK éléments de M
    mergeLocalKernel<<<numParts, CHUNK>>>(
        d_A, sA, d_B, sB, d_M, d_Adiag, d_Bdiag);

    // ---- Vérification ------------------------------------------
    cudaMemcpy(h_M, d_M, sM * sizeof(int), cudaMemcpyDeviceToHost);

    // Vérifier que M est trié et contient 0, 1, 2, 3, ..., sM-1
    int errors = 0;
    for (int i = 0; i < sM; i++) {
        if (h_M[i] != i) {
            if (errors < 10) {
                printf("ERREUR : M[%d] = %d (attendu %d)\n", i, h_M[i], i);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("OK : merge de %d éléments correct !\n", sM);
    } else {
        printf("ECHEC : %d erreurs sur %d éléments\n", errors, sM);
    }

    // ---- Étude du temps d'exécution ----------------------------
    printf("\n=== Étude du temps en fonction de d = |A|+|B| ===\n");

    for (int exp = 16; exp <= 24; exp++) {
        int testSA = 1 << (exp - 1);   // |A| = d/2
        int testSB = 1 << (exp - 1);   // |B| = d/2
        int testSM = testSA + testSB;
        int testParts = (testSM + CHUNK - 1) / CHUNK;

        // Préparer les données
        int *td_A, *td_B, *td_M, *td_Adiag, *td_Bdiag;
        cudaMalloc(&td_A, testSA * sizeof(int));
        cudaMalloc(&td_B, testSB * sizeof(int));
        cudaMalloc(&td_M, testSM * sizeof(int));
        cudaMalloc(&td_Adiag, (testParts + 1) * sizeof(int));
        cudaMalloc(&td_Bdiag, (testParts + 1) * sizeof(int));

        // Remplir avec des données triées
        cudaMemcpy(td_A, h_A, testSA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(td_B, h_B, testSB * sizeof(int), cudaMemcpyHostToDevice);

        // Mesure du temps avec des CUDA events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Étape 1 : partitionnement
        int pb = (testParts + 1 + 255) / 256;
        partitionKernel<<<pb, 256>>>(
            td_A, testSA, td_B, testSB, td_Adiag, td_Bdiag, testParts);

        // Étape 2 : merge local
        mergeLocalKernel<<<testParts, CHUNK>>>(
            td_A, testSA, td_B, testSB, td_M, td_Adiag, td_Bdiag);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        printf("d = 2^%d = %8d | temps = %.3f ms\n", exp, testSM, ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(td_A); cudaFree(td_B); cudaFree(td_M);
        cudaFree(td_Adiag); cudaFree(td_Bdiag);
    }

    // ---- Nettoyage ---------------------------------------------
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
    cudaFree(d_Adiag); cudaFree(d_Bdiag);
    free(h_A); free(h_B); free(h_M);

    return 0;
}