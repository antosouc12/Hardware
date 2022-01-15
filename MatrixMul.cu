

__global__ void MatrixMul(float *M1,float *M2,float *M3, int n){

    int idx= blockIdx.x * blockDim.x + threadIdx.x;

    int l=idx/n;
    int k=idx%n;
    float tmp;
    for (int i=0; i<n;i++){
        tmp+=M1[l+i*n]*M2[i*n+k];
    }
    M3[idx]=tmp;

}
