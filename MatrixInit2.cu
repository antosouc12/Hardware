
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

__global__ void MatrixCoefPrint(float** M, int n){

    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx/n;
    int j = idx%n;

    if (j==(n-1)){
        printf("%f \n" , M[i][j]);
    }
    else{
        printf("%f", M[i][j]);
    }
}


__host__ int main(void){

    int n=5;
    int p=5;

    float **M_h, **M_d;

    M_h=(float**)malloc(n*sizeof(float*));
    for (int m=0;m<n;m++){
        M_h[m]=(float*)malloc(p*sizeof(float));
    }

    int Nthread=1024;
    int Nblock= (n*p+Nthread)/Nthread;

    for (int i=0;i<n;i++){
        for(int j=0;j<p;j++){
            M_h[i][j]=(float)i+j;
        }
    }

    cudaMalloc((void **) &M_d, n*sizeof(float*));

    for(int i=0;i<n;i++){
        cudaMalloc((void **) &(M_d[i]), p*sizeof(float));
        cudaMemcpy(M_d[i],M_h[i],p*sizeof(float),cudaMemcpyHostToDevice);
    }
    
    MatrixCoefPrint<<<Nblock,Nthread>>>(M_d,n);

    for(int i=0;i<n;i++){
        cudaMemcpy(M_h[i],M_d[i],p*sizeof(float),cudaMemcpyDeviceToHost);
    }





    

}

