#include "activation.cu"


__global__ void convolve2d(float* C, float* M, float* K, int n, int ksize){

    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    int x= (idx%(n*n))/n;  // La variable x nous permet de se deplacer selon les lignes 
    int y= idx%n;  //La variable y nous permet de se deplacer selon les colonnes 
    int w = idx/(n*n);  //La variable w nous permet de se deplacer selon la profondeur
    float tmp = float(0);
    
    for(int i=0; i<ksize;i++){
        for (int j=0; j<ksize;j++){
            tmp+=K[(ksize-i-1)*ksize+(ksize-j-1)+w*5*5]*M[(x+i)*n+y+j+w*n*n];
        }
    }
    
    C[idx]=activation(tmp/(n*n));   // Nous passons la valeur dans une fonction d'activation qui est ici la fonction tanh
}


