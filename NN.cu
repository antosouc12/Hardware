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
    printf("You are at cudaMalloc \n");
    cudaMalloc((void **) &M_d,size_f*n*p);
    printf("You are at FillMatrix \n");
    FillMatrix<<<Nblock,Nthread>>>(M_d,n*p);

    cudaMemcpy(raw_data,M_d,n*p*size_f,cudaMemcpyDeviceToHost);

    MatrixPrint(raw_data,32,32);
    
    float *C1_kernel;
    C1_kernel=(float *)malloc(size_f*5*5*6);

    printf("You are at for loop \n");
    for(int i=0;i<6;i++){
        printf("You are at FillMatrix 2 \n");
        FillMatrix<<<Nblock,Nthread>>>(M_d,5*5);
        cudaMemcpy(C1_kernel+i*5*5,M_d,5*5*size_f,cudaMemcpyDeviceToHost);
    }

    MatrixPrint(C1_kernel,5,5);

    // float C1_data[6*28*28];
    // float S1_data[6*14*14];


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


    // for(int i=0;i<28*28*6;i++){
    //     C1_data[i]=(float)i;
    // }

    // for (int i=0;i<14*14*6;i++){
    //     S1_data[i]=(float)i;
    // }
    // printf("You are at memset \n");

    // memset(C1_data, 5, sizeof(C1_data));
    // memset(S1_data, 5, sizeof(S1_data));

    // MatrixPrint(C1_data,28,28);
    // MatrixPrint(S1_data,14,14);

    // cudaMemcpy(C1_data_d,C1_data,28*28*6*size_f,cudaMemcpyHostToDevice);
    // cudaMemcpy(S1_data_d,S1_data,14*14*6*size_f,cudaMemcpyHostToDevice);

    Nblock=28;
    Nthread=28*6;

    //Premiere couche

    // printf("You are at for loop 2 \n");
    // for(int i=0;i<6;i++){
    //     printf("You are at convolve 1\n");
    //     convolve2d<<<Nblock,Nthread>>>(C1_data_d+i*28*28,M_d,C1_kernel_d+i*5*5,32,5);
    //     cudaDeviceSynchronize();
    //     printf("You are at convolve 2\n");
    // }
    
    convolve2d<<<Nblock,Nthread>>>(C1_data_d,M_d,C1_kernel_d,32,5);

    cudaMemcpy(C1_data,C1_data_d,28*28*6*size_f,cudaMemcpyDeviceToHost);
    //cudaFree(C1_data_d);
    
    MatrixPrint(C1_data,28,28);
    
    //Seconde couche 
    
    printf("You are at matmoy \n");

    MatMoy<<<Nblock,Nthread>>>(S1_data_d,C1_data_d,28);

    cudaMemcpy(S1_data,S1_data_d,14*14*6,cudaMemcpyDeviceToHost);

    MatrixPrint(S1_data,14,14);

    printf("You are done \n");

  

    
}