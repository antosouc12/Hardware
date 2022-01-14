#include <curand_kernel.h>


__global__ void FillMatrix(float* M, int n){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state);
    
    if (idx<n){
        M[idx]=curand_uniform_double(&state);
    }
    
}


