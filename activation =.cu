




__global__ void activation(float* M_o, float* M_i, int n){

    int idx= blockDim.x*blockIdx.x + threadIdx.x;

    M_o[idx]=tanh(M_i[idx]);
}