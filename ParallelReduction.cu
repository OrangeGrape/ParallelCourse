#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

__global__ void reduce(float *g_idata, float *g_odata){
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x; 
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();
  
  for(unsigned int s = blockDim.x/2;s>0;s>>=1){
    if(tid<s){
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }
  
  if(tid==0) g_odata[blockIdx.x] = sdata[0];
}

///////////////////////////////////////////////////////////////////////////

void init(float *A, int nA) {
  for (int i=0; i<nA; i++){
      A[i] = (float)rand() / (float)RAND_MAX;
  }
}

float c_summation(float *A, int nA) {
  float sum=A[0];
  for (int i=1; i<nA; i++) {
        sum += A[i];
  }
  return sum;
}

int main() {
  int order = 10;
  int nA = pow(2,order);
  printf("Vector size: %d\n", nA); 
  size_t sizeA = nA*sizeof(float);
  float c_sum,pr_sum;
  float *A,*S;
  A = (float*) malloc(sizeA); 
  S = (float*) malloc(sizeA); 
  
  srand(time(NULL));
  init(A, nA);
  
  float *dA =NULL;
  float *dS =NULL;
  cudaMalloc((void**)&dA,sizeA);
  cudaMalloc((void**)&dS,sizeA);
  cudaMemcpy(dA,A,sizeA,cudaMemcpyHostToDevice);
  //cudaMemcpy(dS,S,sizeA,cudaMemcpyHostToDevice);

  int B = 1024;
  int G = (nA+B-1)/B;
  int smemSize = B*sizeof(float);
  
  float kernel_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  //////////////////////////// Start time record
  reduce<<<G,B,smemSize>>>(dA,dS);
  for(int problemsize=nA/B;problemsize > 1;problemsize/=B){
    reduce<<<G,B,smemSize>>>(dS,dS);
  }
  //////////////////////////// Stop time record
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernel_time,start,stop);
  printf("Kernel time = %lf s\n", kernel_time*0.001);
  
  cudaMemcpy(S,dS,sizeA,cudaMemcpyDeviceToHost);
  pr_sum = S[0]; 
  
  clock_t begin, end;
  begin = clock();
  c_sum = c_summation(A,nA);
  end = clock();
  double cpu_time = (double) (end-begin)/ CLOCKS_PER_SEC;
  printf("CPU time = %lf s\n", cpu_time);
  
  printf("Parallel reduction sum result is: %f \n",pr_sum);
  printf("c function sum result is: %f \n", c_sum);

  free(A);
  cudaFree(dA);
  
  return 0;
}
