#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void Matmul(float *A,float *B,float *C,int wA,int wB,int hC){
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  int k;
  if ( i<wB && j<hC ){
    for(k=0;k<wA;k++){
      
    }
  }
}


void init(float *A, int wA, int hA) {
  for (int h=0; h<hA; h++)
    for (int w=0; w<wA; w++)
      A[w+h*wA] = (float)rand() / (float)RAND_MAX;
}
void compute(float *A, float *B, float *C,
             int wA, int hA, int wB) {
  for (int h=0; h<hA; h++) {
    for (int w=0; w<wB; w++) {
      float temp = 0.0f;
      for (int i=0; i<wA; i++)
        temp += A[i+h*wA] * B[w+i*wB];
      C[w+h*wB] = temp;
    }
  }
}

int main() {
  clock_t start, stop;
  int w, h, i, iter, max_iter = 10;
  //int wA = 2, hA = 2, wB = 2, hB = 2;
  int wA = 320, hA = 320, wB = 640, hB = 320;
  size_t sizeA = wA*hA*sizeof(float);
  size_t sizeB = wB*hB*sizeof(float);
  size_t sizeC = hA*wB*sizeof(float);
  float *A, *B, *C;
  A = (float*) malloc(sizeA);
  B = (float*) malloc(sizeB);
  C = (float*) malloc(sizeC);
  
  float *dA =NULL;
  float *dB =NULL;
  float *dC =NULL;
  cudaMalloc((void**)&dA,sizeA);
  cudaMalloc((void**)&dA,sizeB);
  cudaMalloc((void**)&dA,sizeC);
  
  // seed random number generator
  srand(time(NULL));

  // initialize A
  init(A, wA, hA);
  init(B, wB, hB);

  //prepare memory in cuda
  float *dA =NULL;
  float *dB =NULL;
  float *dC =NULL;
  cudaMalloc((void**)&dA,sizeA);
  cudaMalloc((void**)&dA,sizeB);
  cudaMalloc((void**)&dA,sizeC);
  
  cudMemcpy(dA,A,sizeA,cudaMemcpyHostToDevice);
  cudMemcpy(dB,B,sizeB,cudaMemcpyHostToDevice);
  
  // compute C
  
start = clock();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&start);
  Matmul<<<G,B>>>(A,B,C,wB,hA);
  printf("CPU time = %lf\n", cpu_time);

/*
  // output A
  printf("A = \n");
  for (h=0; h<hA; h++) {
    for (w=0; w<wA; w++)
      printf("%5.2f\t", A[w+h*wA]);
    printf("\n");
  }

  // output B
  printf("B = \n");
  for (h=0; h<hB; h++) {
    for (w=0; w<wB; w++)
      printf("%5.2f\t", B[w+h*wB]);
    printf("\n");
  }

  // output C
  printf("C = \n");
  for (h=0; h<hA; h++) {
    for (w=0; w<wB; w++)
      printf("%5.2f\t", C[w+h*wB]);
    printf("\n");
  }
*/
  return 0;
}
