

__global__ void MatrixAdd(float *Mat1,float *Mat2,float *Mat3, int n, int p){

    int idx= blockIdx.x* blockDim.x+ threadIdx.x;

    if (idx<n*p){
        Mat3[idx]=Mat2[idx]+Mat1[idx];
    }
}
