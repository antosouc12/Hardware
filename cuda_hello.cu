#include <stdio.h> 

__global__ void cuda_hello(){
    printf("Hello World, from GPU! \n");
    printf("You are in the %d Block \n", blockIdx.x);
    printf("You are in the %d thread \n", threadIdx.x);

    
}

int main(){
    //__global__ printf("The dimensions of your block is %d", blockDim.x); 
    cuda_hello<<<2,6>>>();

    cudaDeviceSynchronize();
    return 0;
}