#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// Merge two sorted arrays A and B in M.
// Thread of id i find M[i] using a binary search
// of the merge path on the diagonal {(x,y) | x + y = i}.
__global__ void mergeSmall_k(int* A, int sA,
                              int* B, int sB,
                              int* M) {


    // We assume num_block = 1
    int i = threadIdx.x;
    if (i >= sA + sB) return;

    int Kx, Ky, Px, Py;
    if (i > sA) {
        Kx = i - sA;  
        Ky = sA;

        Px = sA;
        Py = i - sA;

    } else {
        Kx = 0;
        Ky = i;

        Px = i;
        Py = 0;
    }

    // Q is on the merge path iff those 2 conditions are verified :
    //   (1) A[Qy] > B[Qx-1]  : B’s elements are correctly placed before A (not too far right)
    //   (2) A[Qy-1] <= B[Qx] : A’s elements are correctly placed before B (not too far down)
    while (true) {
        int offset = abs(Ky - Py) / 2;
        int Qx = Kx + offset;
        int Qy = Ky - offset;

        if (Qy >= 0 && Qx <= sB &&
            (Qy == sA || Qx == 0 || A[Qy] > B[Qx - 1])) {

            if (Qx == sB || Qy == 0 || A[Qy - 1] <= B[Qx]) {
                // Q is on the merge path, we can now decide the value of M[i]
                if (Qy < sA && (Qx == sB || A[Qy] <= B[Qx])){
                    M[i] = A[Qy];
                } else {
                    M[i] = B[Qx];
                }
                return;
            } else {
                // Q is too far down, and need to be higher (hence more on the right) 
                // We update the low point of the diagonal to keep only the top right part
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        } else {
            // Q is too far right, and need to be more on the left (hence lower) 
            // We update the high point of the diagonal to keep only the bottom left part
            Px = Qx - 1;
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
