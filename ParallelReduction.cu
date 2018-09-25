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
  int order = 20;
  int nA = pow(2,order);
  printf("nA: %d\n", nA); 
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
  
  float cpu_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  //////////////////////////// Start time record
  reduce<<<G,B,smemSize>>>(dA,dS);
  int count = 0;
  for(int problemsize=nA/B;problemsize >= B;problemsize/=B){
    reduce<<<G,B,smemSize>>>(dS,dS);
    count ++;
  }
  printf("count: %d\n",count);
  //////////////////////////// Stop time record
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_time,start,stop);
  printf("CPU time = %lf s\n", cpu_time*0.001);
  
  cudaMemcpy(S,dS,sizeA,cudaMemcpyDeviceToHost);
  for(int i=0;i<1024;i++){
    printf("%.2f \t",S[i]);
  }
  printf("\n");

  c_sum = c_summation(A,nA);
  pr_sum = S[0]; 
  
  
  printf("c function sumresult is: %f \n", c_sum);
  printf("Parallel reduction sumresult is: %f \n",pr_sum);

  free(A);
  cudaFree(dA);
  
  return 0;
}
