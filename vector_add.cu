#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c){

    int idx=blockIdx.x *blockDim.x + threadIdx.x;
    
    c[idx]=a[idx]+b[idx];
    //printf("%d \n", threadIdx.x);
    //printf("%d \n", blockIdx.x);


}

__host__ int main(void){

    int N=10000000;
    int size= sizeof(float);
    float *a_h,*b_h,*c_h,*a_d,*b_d,*c_d ;
    int Nthread=1024;
    int Nblock= (N+Nthread)/Nthread;

    a_h=(float*)malloc(N*size);
    b_h=(float*)malloc(N*size);
    c_h=(float*)malloc(N*size);

    for (int i=0;i<N;i++){
        a_h[i]=(float) 3;
        b_h[i]=(float) 5;
    }

    cudaMalloc((void **) &a_d, N*size);
    cudaMalloc((void **) &b_d, N*size);
    cudaMalloc((void **) &c_d, N*size);

    cudaMemcpy(a_d,a_h,N*size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h,N*size,cudaMemcpyHostToDevice);
    cudaMemcpy(c_d,c_h,N*size,cudaMemcpyHostToDevice);

    vector_add<<<Nblock,Nthread>>>(a_d,b_d,c_d);

    cudaMemcpy(a_h,a_d,N*size,cudaMemcpyDeviceToHost);
    cudaMemcpy(b_h,b_d,N*size,cudaMemcpyDeviceToHost);
    cudaMemcpy(c_h,c_d,N*size,cudaMemcpyDeviceToHost);

    // cudaFree(a_d);
    // cudaFree(b_d);
    // cudaFree(c_d);

    // free(a_h);
    // free(b_h);
    // free(c_h);

    printf("%f \n", c_h[0]);
    
}