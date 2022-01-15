
__global__ void MatMoy(float* M_o,float* M_i, int n){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    int m=n/2;
    int x=(idx%(m*m))/m;
    int y=idx%(m);
    int w=idx/(m*m);

    M_o[idx]= (M_i[2*(x*n+y)+w*n*n]+M_i[2*(x*n+y)+1+w*n*n]+M_i[(2*x+1)*n+2*y+w*n*n]+M_i[(2*x+1)*n+2*y+1+w*n*n])/4;

}
