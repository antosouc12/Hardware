// #include "MatrixPrint.cu"

__global__ void MatMoy(float* M_o,float* M_i, int n){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    int m=n/2;
    int x=(idx%(m*m))/m;
    int y=idx%(m);
    int w=idx/(m*m);

    M_o[idx]= (M_i[2*(x*n+y)+w*n*n]+M_i[2*(x*n+y)+1+w*n*n]+M_i[(2*x+1)*n+2*y+w*n*n]+M_i[(2*x+1)*n+2*y+1+w*n*n])/4;

    if(idx==4){
        printf("M_i[2*(x*n+y)+w*n*n] = %f \n", M_i[2*(x*n+y)+w*n*n]);
        printf("M_i[2*(x*n+y)+1+w*n*n] = %f \n", M_i[2*(x*n+y)+1+w*n*n]);
        printf("M_i[2*(x*n+y)+n+w*n*n] = %f \n", M_i[2*(x*n+y)+n+w*n*n]);
        printf("M_i[2*(x*n+y)+n+1+w*n*n] = %f \n", M_i[2*(x*n+y)+n+1+w*n*n]);
        printf("x = %d \n", x);
        printf("y = %d \n", y);
    }
}


// __host__ int main(){

//     int n=8;
//     int p=8;
//     int h=6;

//     float * M_h1, *M_h2, *M_d1, *M_d2;

//     M_h1=(float*)malloc(n*p*h*sizeof(float));
//     M_h2=(float*)malloc(n*p*h*sizeof(float)/4);

//     cudaMalloc((void **) &M_d1, n*p*h*sizeof(float));
//     cudaMalloc((void **) &M_d2, n*p*h*sizeof(float)/4);

//     for (int i=0;i<n*p*h;i++){
//         M_h1[i]=(float) i;
//     }

//     MatrixPrint(M_h1,n,p);
//     MatrixPrint(M_h2,n/2,p/2);

//     cudaMemcpy(M_d1,M_h1,n*p*h*sizeof(float),cudaMemcpyHostToDevice);

//     int Nthread= p*h;
//     int Nblock = n;

//     MatMoy<<<Nblock,Nthread>>>(M_d2,M_d1,8);

//     cudaMemcpy(M_h2,M_d2,n*p*h*sizeof(float)/4,cudaMemcpyDeviceToHost);

//     MatrixPrint(M_h2,n/2,p/2);
// }

