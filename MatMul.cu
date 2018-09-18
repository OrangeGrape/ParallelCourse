#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//kernel
__global__ void Matmul(float *A,float *B,float *C,int wA,int wC,int hC){
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int j = blockDim.y*blockIdx.y+threadIdx.y;
  int k;
  float tmp = 0.0f;

  for(k=0;k<wA;k++){
    tmp += A[i+j*wA] * B[i+j*wB];
  }
  C[i+j*wC] = tmp;

}

//C function
void init(float *A, int wA, int hA) {
  for (int h=0; h<hA; h++)
    for (int w=0; w<wA; w++)
      A[w+h*wA] = (float)rand() / (float)RAND_MAX;
}
void compute(float *A, float *B, float *C,int wA, int hA, int wB) {
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
  float cpu_time;
  int w, h, i, iter, max_iter = 10;
  int wA = 320, hA = 320, wB = 640, hB = 320;
  int wC = wB, hC = hA;
  size_t sizeA = wA*hA*sizeof(float);
  size_t sizeB = wB*hB*sizeof(float);
  size_t sizeC = hA*wB*sizeof(float);
  float *A, *B, *C;
  A = (float*) malloc(sizeA);
  B = (float*) malloc(sizeB);
  C = (float*) malloc(sizeC);
  
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
  
  //coppy input value from host to cuda
  cudMemcpy(dA,A,sizeA,cudaMemcpyHostToDevice);
  cudMemcpy(dB,B,sizeB,cudaMemcpyHostToDevice);
  
  //separate tasks
  dim3 G(wC/32,hC/32);
  dim3 B(32,32);
  
  // compute matric C
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  for (iter=0; iter<max_iter; iter++){
    Matmul<<<G,B>>>(A,B,C,wB,hA);
  }
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_time,start,stop);
  printf("CPU time = %lf s\n", cpu_time*0.001/max_iter);
  
  //coppy output value from cuda to host
  cudMemcpy(C,dC,sizeC,cudaMemcpyDeviceToHost);

  //result check
///*
  float *Check;
  Check = (float*) malloc(sizeC);
  compute(A, B, Check, wA, hA, wB);
  float sum = 0.0f;
  for (int h=0; h<hC; h++) {
    for (int w=0; w<wC; w++) {
      sum += C[w+h*wC]-Check[w+h*wC];
    }
  }
  printf("Check result %f (should be zero)", sum);
  free(Check);
//*/
  //free memory
  free(A);
  free(B);
  free(C);
  
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
