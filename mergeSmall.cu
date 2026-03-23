#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ================================================================
// KERNEL : mergeSmall_k
// Fusionne deux tableaux triés A et B dans M.
// Chaque thread s'occupe d'UNE case de M (= une diagonale).
// Contrainte : |A| + |B| <= 1024 (max threads par bloc CUDA)
// Lancement : mergeSmall_k<<<1, sA+sB>>>(...)
//
// Algorithme : Merge Path (cf. [6] Green, McColl, Bader 2012)
// On traite le merge comme un chemin dans une grille |A| × |B|.
// Chaque thread k trouve l'intersection du chemin avec la
// diagonale k par recherche binaire, puis écrit M[k].
// ================================================================
__global__ void mergeSmall_k(int* A, int sA,
                              int* B, int sB,
                              int* M) {

    // ---- BLOC 1 : qui suis-je ? --------------------------------
    // threadIdx.x = numéro de ce thread dans le bloc (0, 1, 2, ...)
    // i = numéro de ma diagonale = ma position dans M
    // Thread 0 → M[0], Thread 1 → M[1], etc.
    int i = threadIdx.x;
    int n = sA + sB;  // taille totale de M

    // Sécurité : si on a lancé trop de threads, ceux en trop s'arrêtent
    if (i >= n) return;

    // ---- BLOC 2 : bornes de ma diagonale -----------------------
    // La grille a |A| lignes et |B| colonnes.
    // Chaque diagonale i a un point bas K et un point haut P.
    //   Kx/Px = colonne dans la grille = index dans B
    //   Ky/Py = ligne   dans la grille = index dans A
    //
    // Deux cas (cf. Algorithme 2 du sujet) :
    //
    //   i <= sA  →  diagonale ne déborde pas en bas
    //               K = (0,  i)     sur le bord GAUCHE
    //               P = (i,  0)     sur le bord HAUT
    //
    //   i >  sA  →  diagonale déborde en bas
    //               K = (i-sA, sA)  sur le bord BAS
    //               P = (sA, i-sA)  sur le bord DROIT/HAUT
    //
    // ATTENTION : dans le cas i > sA, les coordonnées de P
    // dépendent de sA (pas de sB). C'est bien |A| qui apparaît
    // dans la formule P = (|A|, i − |A|) de l'Algorithme 2.
    int Kx, Ky, Px, Py;

    if (i > sA) {
        Kx = i - sA;   Ky = sA;      // K sur le bord bas
        Px = sA;        Py = i - sA;  // P : Algorithme 2 → P = (|A|, i−|A|)
    } else {
        Kx = 0;   Ky = i;   // K sur le bord gauche
        Px = i;   Py = 0;   // P sur le bord haut
    }

    // ---- BLOC 3 : recherche binaire ----------------------------
    // On cherche le point Q sur le merge path (= la frontière 0/1).
    // À chaque itération on prend le milieu de l'intervalle [K, P]
    // sur la diagonale et on teste si Q est sur le chemin.
    //
    // Condition "Q est sur le chemin" (deux tests) :
    //   1) A[Qy] > B[Qx-1]  (ou bord)  → Q est sous ou sur la frontière
    //   2) A[Qy-1] <= B[Qx]  (ou bord)  → Q est au-dessus ou sur la frontière
    // Si les deux sont vrais → Q est exactement sur le chemin.
    while (1) {

        // Milieu de l'intervalle [K, P] sur la diagonale
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;   // colonne de Q (index dans B)
        int Qy = Ky - offset;   // ligne   de Q (index dans A)

        // --- Test 1 : Q n'est pas trop bas ? ---
        if (Qy >= 0 && Qx <= sB &&
            (Qy == sA || Qx == 0 || A[Qy] > B[Qx - 1])) {

            // --- Test 2 : Q n'est pas trop haut ? ---
            if (Qx == sB || Qy == 0 || A[Qy - 1] <= B[Qx]) {

                // ---- TROUVÉ : Q est sur le merge path ----
                // Le chemin passe par Q. On décide quoi écrire dans M[i] :
                //   - si on peut prendre dans A ET A[Qy] <= B[Qx] → M[i] = A[Qy]
                //   - sinon → M[i] = B[Qx]
                if (Qy < sA && (Qx == sB || A[Qy] <= B[Qx])) {
                    M[i] = A[Qy];  // le chemin descend → on prend A
                } else {
                    M[i] = B[Qx];  // le chemin va à droite → on prend B
                }
                return;  // travail terminé pour ce thread

            } else {
                // Q est trop haut → on remonte K (on cherche plus bas)
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else {
            // Q est trop bas → on descend P (on cherche plus haut)
            Px = Qx - 1;
            Py = Qy + 1;
        }
    }
}

// ================================================================
// MAIN — deux tests de validation
// ================================================================
int main() {

    // ============================================================
    // TEST 1 : petit exemple (sA = sB à peu près)
    // ============================================================
    printf("=== TEST 1 : A={1,5,6,9}, B={2,4,7} ===\n");
    {
        int A[] = {1, 5, 6, 9};
        int B[] = {2, 4, 7};
        int sA  = 4;
        int sB  = 3;
        int sM  = sA + sB;   // = 7
        int M[7] = {0};

        // Pointeurs GPU (préfixe d_ = device)
        int *d_A, *d_B, *d_M;

        // Allocation mémoire sur le GPU
        cudaMalloc(&d_A, sA * sizeof(int));
        cudaMalloc(&d_B, sB * sizeof(int));
        cudaMalloc(&d_M, sM * sizeof(int));

        // Copie CPU → GPU
        cudaMemcpy(d_A, A, sA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sB * sizeof(int), cudaMemcpyHostToDevice);

        // Lancement : 1 bloc, sM threads (un thread par élément de M)
        mergeSmall_k<<<1, sM>>>(d_A, sA, d_B, sB, d_M);

        // Copie GPU → CPU
        cudaMemcpy(M, d_M, sM * sizeof(int), cudaMemcpyDeviceToHost);

        // Attendu : 1 2 4 5 6 7 9
        printf("M = ");
        for (int i = 0; i < sM; i++) printf("%d ", M[i]);
        printf("\n");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_M);
    }

    // ============================================================
    // TEST 2 : tailles très différentes (sA << sB)
    // Ce cas teste le calcul de P quand i > sA.
    // ============================================================
    printf("\n=== TEST 2 : A={10,20,30}, B={5,15,18,22,25,35,40} ===\n");
    {
        int A[] = {10, 20, 30};
        int B[] = {5, 15, 18, 22, 25, 35, 40};
        int sA = 3;
        int sB = 7;
        int sM = sA + sB;   // = 10
        int M[10] = {0};

        int *d_A, *d_B, *d_M;
        cudaMalloc(&d_A, sA * sizeof(int));
        cudaMalloc(&d_B, sB * sizeof(int));
        cudaMalloc(&d_M, sM * sizeof(int));

        cudaMemcpy(d_A, A, sA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sB * sizeof(int), cudaMemcpyHostToDevice);

        mergeSmall_k<<<1, sM>>>(d_A, sA, d_B, sB, d_M);

        cudaMemcpy(M, d_M, sM * sizeof(int), cudaMemcpyDeviceToHost);

        // Attendu : 5 10 15 18 20 22 25 30 35 40
        printf("M = ");
        for (int i = 0; i < sM; i++) printf("%d ", M[i]);
        printf("\n");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_M);
    }

    return 0;
}