#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "MatrixInit.cu"
#include "MatrixConv.cu"
#include "MatrixMoy.cu"
#include "MatrixPrint.cu"

__host__ int main(){    


    int n=32;
    int p=32;
    int size_f=sizeof(float);
    int Nthread=1024;
    int Nblock=(n*p+Nthread)/Nthread;

    float *raw_data;
    raw_data=(float*)malloc(n*p*sizeof(float));
    float *M_d;
    cudaMalloc((void **) &M_d,size_f*n*p);
    FillMatrix<<<Nblock,Nthread>>>(M_d,n*p);

    cudaMemcpy(raw_data,M_d,n*p*size_f,cudaMemcpyDeviceToHost);

    MatrixPrint(raw_data,32,32);
    
    float *C1_kernel;
    C1_kernel=(float *)malloc(size_f*5*5*6);

    for(int i=0;i<6;i++){
        FillMatrix<<<Nblock,Nthread>>>(M_d,5*5);
        cudaMemcpy(C1_kernel+i*5*5,M_d,5*5*size_f,cudaMemcpyDeviceToHost);
    }

    MatrixPrint(C1_kernel,5,5);

    float *C1_data;
    float *S1_data;
    C1_data=(float*)malloc(28*28*6*sizeof(float));
    S1_data=(float*)malloc(14*14*6*sizeof(float));
    
    float *C1_data_d;
    float *C1_kernel_d;
    float *S1_data_d;
    cudaMalloc((void **) &C1_kernel_d, 5*5*6*sizeof(float));
    cudaMalloc((void **) &C1_data_d, 28*28*6*sizeof(float));
    cudaMalloc((void **) &S1_data_d, 14*14*6*sizeof(float));

    cudaMemcpy(C1_kernel_d,C1_kernel, 5*5*6*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(M_d,raw_data,32*32*6*sizeof(float),cudaMemcpyHostToDevice);

    Nblock=28;
    Nthread=28*6;

    //Premiere couche
    
    convolve2d<<<Nblock,Nthread>>>(C1_data_d,M_d,C1_kernel_d,32,5);

    cudaMemcpy(C1_data,C1_data_d,28*28*6*size_f,cudaMemcpyDeviceToHost);
    
    MatrixPrint(C1_data,28,28);
    
    //Seconde couche 

    MatMoy<<<Nblock,Nthread>>>(S1_data_d,C1_data_d,28);

    cudaMemcpy(S1_data,S1_data_d,14*14*6,cudaMemcpyDeviceToHost);

    MatrixPrint(S1_data,14,14);

}
