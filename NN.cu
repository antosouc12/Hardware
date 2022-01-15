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
`   
    float *raw_data;    // C'est la matrice dans laquelle se trouvera notre image d'origine dans le CPU
    raw_data=(float*)malloc(n*p*sizeof(float));
    float *M_d;     // C'est la matrice dans laquelle se trouvera notre image d'origine dans le GPU
    cudaMalloc((void **) &M_d,size_f*n*p);
    FillMatrix<<<Nblock,Nthread>>>(M_d,n*p); // On remplit la matrice dans le GPU

    cudaMemcpy(raw_data,M_d,n*p*size_f,cudaMemcpyDeviceToHost); 

    MatrixPrint(raw_data,32,32);
    
    float *C1_kernel;   //C'est la matrice dans laquelle nos 6 kernels se trouveront dans le CPU
    C1_kernel=(float *)malloc(size_f*5*5*6);

    for(int i=0;i<6;i++){
        FillMatrix<<<Nblock,Nthread>>>(M_d,5*5);  //Nous remplissons la matrice dans le GPU cependant nous faisons une boucle dans le CPU 6 fois pour la remplir.
                                                  // Nous aurions pu faire tout directement dans le GPU mais cela facilite le code 
        cudaMemcpy(C1_kernel+i*5*5,M_d,5*5*size_f,cudaMemcpyDeviceToHost);
    }

    MatrixPrint(C1_kernel,5,5);

    float *C1_data; // C'est la matrice dans laquelle se trouvent les valeurs apres la premiere couche de convolution dans le CPU
    float *S1_data; // C'est la matrice dans laquelle se trouvent les valeurs apres la premiere couche de moyenne pooling dans le CPU
    C1_data=(float*)malloc(28*28*6*sizeof(float));
    S1_data=(float*)malloc(14*14*6*sizeof(float));
    
    float *C1_data_d;  // C'est la matrice dans laquelle se trouvent les valeurs apres la premiere couche de convolution dans le GPU
    float *C1_kernel_d;  //C'est la matrice dans laquelle nos 6 kernels se trouveront dans le GPU
    float *S1_data_d;  // C'est la matrice dans laquelle se trouvent les valeurs apres la premiere couche de moyenne pooling dans le GPU
    cudaMalloc((void **) &C1_kernel_d, 5*5*6*sizeof(float));
    cudaMalloc((void **) &C1_data_d, 28*28*6*sizeof(float));
    cudaMalloc((void **) &S1_data_d, 14*14*6*sizeof(float));

    cudaMemcpy(C1_kernel_d,C1_kernel, 5*5*6*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(M_d,raw_data,32*32*6*sizeof(float),cudaMemcpyHostToDevice);

    Nblock=28;
    Nthread=28*6;

    //Premiere couche
    
    convolve2d<<<Nblock,Nthread>>>(C1_data_d,M_d,C1_kernel_d,32,5); //On fait attention de bien utiliser les matrices GPUs lorsque l'on appelle les fonctions __global__ ou __device__

    cudaMemcpy(C1_data,C1_data_d,28*28*6*size_f,cudaMemcpyDeviceToHost); //On copie bien l'information des matrices GPU vers les matrices CPU
    
    MatrixPrint(C1_data,28,28); //On fait attention de bien utiliser les matrices GPUs lorsque l'on appelle les fonctions __host__
    
    //Seconde couche 

    MatMoy<<<Nblock,Nthread>>>(S1_data_d,C1_data_d,28);

    cudaMemcpy(S1_data,S1_data_d,14*14*6,cudaMemcpyDeviceToHost);

    MatrixPrint(S1_data,14,14);

}
