#include "activation.cu"


__global__ void convolve2d(float* C, float* M, float* K, int n, int ksize){

    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    int x= (idx%(n*n))/n;
    int y= idx%n; 
    int w = idx/(n*n);
    float tmp = float(0);
    
    for(int i=0; i<ksize;i++){
        for (int j=0; j<ksize;j++){
            tmp+=K[(ksize-i-1)*ksize+(ksize-j-1)+w*5*5]*M[(x+i)*n+y+j];
        }
    }
    
    C[idx]=activation(tmp/(n*n));
}


