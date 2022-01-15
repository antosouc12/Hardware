#include <curand_kernel.h> //Curand kernel est une library qui nous permet de generer des valeurs aleatoires a l'interieur du GPU


__global__ void FillMatrix(float* M, int n){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state);       //Ces deux lignes permettent d'initer le generateur aleatoire
    
    if (idx<n){
        M[idx]=curand_uniform_double(&state);
    }
    
}


