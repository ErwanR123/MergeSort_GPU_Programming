#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Fusionne deux tableaux triés A (taille sA) et B (taille sB) dans M.
// Un thread par élément de M, lancement : mergeSmall_k<<<1, sA+sB>>>
// Algorithme Merge Path (Green, McColl, Bader 2012) : chaque thread i
// cherche par dichotomie l'intersection du merge path avec la diagonale i.
__global__ void mergeSmall_k(int* A, int sA,
                              int* B, int sB,
                              int* M) {

    int i = threadIdx.x;
    if (i >= sA + sB) return;

    // Bornes K (bas-gauche) et P (haut-droite) de la diagonale i dans la grille.
    // x = index dans B (colonne), y = index dans A (ligne).
    // Si i > sA, la diagonale déborde par le bas : K est sur le bord bas.
    int Kx, Ky, Px, Py;
    if (i > sA) {
        Kx = i - sA;  Ky = sA;     // bord bas
        Px = sA;      Py = i - sA; // cf. Algorithme 2 : P = (|A|, i-|A|)
    } else {
        Kx = 0;  Ky = i;  // bord gauche
        Px = i;  Py = 0;  // bord haut
    }

    // Recherche binaire du merge path sur [K, P].
    // Q est sur le chemin si et seulement si :
    //   (1) A[Qy] > B[Qx-1]  : le dernier B pris est bien avant le prochain A
    //   (2) A[Qy-1] <= B[Qx] : le dernier A pris est bien avant le prochain B
    while (1) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= sB &&
            (Qy == sA || Qx == 0 || A[Qy] > B[Qx - 1])) {

            if (Qx == sB || Qy == 0 || A[Qy - 1] <= B[Qx]) {
                // Q est sur le merge path : on écrit M[i]
                M[i] = (Qy < sA && (Qx == sB || A[Qy] <= B[Qx])) ? A[Qy] : B[Qx];
                return;
            } else {
                Kx = Qx + 1;  // Q trop haut, on remonte K
                Ky = Qy - 1;
            }
        } else {
            Px = Qx - 1;  // Q trop bas, on descend P
            Py = Qy + 1;
        }
    }
}

int main() {

    // TEST 1 : sA > sB
    printf("=== TEST 1 : A={1,5,6,9}, B={2,4,7} ===\n");
    {
        int A[] = {1, 5, 6, 9};
        int B[] = {2, 4, 7};
        int sA = 4, sB = 3, sM = sA + sB;
        int M[7] = {0};
        int *d_A, *d_B, *d_M;

        cudaMalloc(&d_A, sA * sizeof(int));
        cudaMalloc(&d_B, sB * sizeof(int));
        cudaMalloc(&d_M, sM * sizeof(int));
        cudaMemcpy(d_A, A, sA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sB * sizeof(int), cudaMemcpyHostToDevice);

        mergeSmall_k<<<1, sM>>>(d_A, sA, d_B, sB, d_M);
        cudaMemcpy(M, d_M, sM * sizeof(int), cudaMemcpyDeviceToHost);

        printf("M = ");
        for (int i = 0; i < sM; i++) printf("%d ", M[i]);
        printf("\n(attendu : 1 2 4 5 6 7 9)\n");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
    }

    // TEST 2 : sA << sB, vérifie le calcul de P pour i > sA
    printf("\n=== TEST 2 : A={10,20,30}, B={5,15,18,22,25,35,40} ===\n");
    {
        int A[] = {10, 20, 30};
        int B[] = {5, 15, 18, 22, 25, 35, 40};
        int sA = 3, sB = 7, sM = sA + sB;
        int M[10] = {0};
        int *d_A, *d_B, *d_M;

        cudaMalloc(&d_A, sA * sizeof(int));
        cudaMalloc(&d_B, sB * sizeof(int));
        cudaMalloc(&d_M, sM * sizeof(int));
        cudaMemcpy(d_A, A, sA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sB * sizeof(int), cudaMemcpyHostToDevice);

        mergeSmall_k<<<1, sM>>>(d_A, sA, d_B, sB, d_M);
        cudaMemcpy(M, d_M, sM * sizeof(int), cudaMemcpyDeviceToHost);

        printf("M = ");
        for (int i = 0; i < sM; i++) printf("%d ", M[i]);
        printf("\n(attendu : 5 10 15 18 20 22 25 30 35 40)\n");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_M);
    }

    return 0;
}
