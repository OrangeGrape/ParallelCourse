/* Solving the 2D acoustic wave equation using explicit finite
 * difference method
 * Copyright 2018 Chaiwoot Boonyasiriwat. All rights reserved.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//Kernel
__global__ void Wcalculate(float *u0,float *u1,float *u2,float C2,int nx,int ny){
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  int j=blockDim.y*blockIdx.y+threadIdx.y;
  if( i>0 && i<nx-1 && j>0 && j<ny-1){
    u2[i+j*nx] = (2.0f-4.0f*C2)*u1[i+j*nx] - u0[i+j*nx] + C2*(u1[(i+1)+j*nx]+u1[(i-1)+j*nx] + u1[i+(j+1)*nx]+u1[i+(j-1)*nx]);
  }
}

__global__ void Wupdate(float *u0,float *u1,float *u2,int nx,int ny){
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  int j=blockDim.y*blockIdx.y+threadIdx.y;
  if( i < nx-1 && j < ny-1){
    u0[i+j*nx] = u1[i+j*nx];
    u1[i+j*nx] = u2[i+j*nx];
  }
}


int main() {
//allocate parameter
  size_t size;
  clock_t start, stop;
  int nx, ny, nt, ix, iy, it, indx;
  float v, dx, dt, C, C2, xmax, ymax, a;
  float *u0_h, *u1_h, *u2_h;
//set value  
  xmax = 1.0f;
  ymax = 1.0f;
  nx = 201;
  ny = 201;
  v = 0.1f;
  dx = xmax/(nx-1);
  dt = 0.035f;
  C = v*dt/dx;
  C2 = C*C;
  nt = 1000;
  a = 1000.0;
  size = nx*ny*sizeof(float);
  
  u0_h = (float*) malloc(size);
  u1_h = (float*) malloc(size);
  u2_h = (float*) malloc(size);

  float *u0_cu = NULL;
  cudaMalloc((void**)&u0_cu,size);
  float *u1_cu = NULL;
  cudaMalloc((void**)&u1_cu,size);
  float *u2_cu = NULL;
  cudaMalloc((void**)&u2_cu,size);

//initial u0 u1  
  for (iy=0; iy<ny; iy++) {
    float yy = iy*dx - 0.5*ymax;
    for (ix=0; ix<nx; ix++) {
      indx = ix+iy*nx;
      float xx = ix*dx - 0.5*xmax;
      u0_h[indx] = exp(-a*(pow(xx,2)+pow(yy,2)));
      u1_h[indx] = u0_h[indx]; 
      u2_h[indx] = 0;
    }
  }


//coppy u0 -> u0_cu, u1 -> u1_cu
  cudaMemcpy(u0_cu, u0_h, size,cudaMemcpyHostToDevice);
  cudaMemcpy(u1_cu, u1_h, size,cudaMemcpyHostToDevice);
  cudaMemcpy(u2_cu, u2_h, size,cudaMemcpyHostToDevice);
  
//start wave calculation looping time
  start = clock();
  dim3 G(nx/32 +1,ny/32 +1);
  dim3 B(32,32);

  for (it=0; it<nt; it++) {
    // advance wavefields at inner nodes
    Wcalculate<<<G,B>>>(u0_cu, u1_cu, u2_cu, C2, nx, ny);
    // update
    Wupdate<<<G,B>>>(u0_cu, u1_cu, u2_cu, nx, ny);
  }
  
  cudaMemcpy(u2_h, u2_cu, size,cudaMemcpyDeviceToHost);
  
  stop = clock();
//end calculation

  double cpu_time = (double) (stop-start) / CLOCKS_PER_SEC;
  printf("CPU time = %lf s\n", cpu_time);

// output the final snapshot
  FILE *file = fopen("u.dat","w");
  fwrite(u2_h, sizeof(float), nx*ny, file);
  fclose(file);

// Free memory
  free(u0_h);
  free(u1_h);
  free(u2_h);
  
  cudaFree(u1_cu);
  cudaFree(u1_cu);
  cudaFree(u2_cu);

  return 0;
}
