/* Solving the 2D acoustic wave equation using explicit finite
 * difference method
 * Copyright 2018 Chaiwoot Boonyasiriwat. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
int main() {
  size_t size;
  clock_t start, stop;
  int nx, ny, nt, ix, iy, it, indx;
  float v, dx, dt, C, C2, xmax, ymax, a;
  float *u0, *u1, *u2;
  xmax = 1.0f;
  ymax = 1.0f;
  nx = 201;
  ny = 201;
  v = 0.1f;
  dx = xmax/(nx-1);
  dt = 0.035f;
  C = v*dt/dx;
  nt = 1000;
  a = 1000.0;
  size = nx*ny*sizeof(float);
  u0 = (float*) malloc(size);
  u1 = (float*) malloc(size);
  u2 = (float*) malloc(size);
  for (iy=0; iy<ny; iy++) {
    float yy = iy*dx - 0.5*ymax;
    for (ix=0; ix<nx; ix++) {
      indx = ix+iy*nx;
      float xx = ix*dx - 0.5*xmax;
      u0[indx] = exp(-a*(pow(xx,2)+pow(yy,2)));
      u1[indx] = u0[indx];
    }
  }
  C = dt*v/dx;
  C2 = C*C;
  start = clock();
  for (it=0; it<nt; it++) {
    // advance wavefields at inner nodes
    for (iy=1; iy<ny-1; iy++) {
      for (ix=1; ix<nx-1; ix++) {
        indx = ix+iy*nx;
        u2[indx] = (2.0f-4.0f*C2)*u1[indx] - u0[indx]
          + C2*(u1[(ix+1)+iy*nx]+u1[(ix-1)+iy*nx]
              +u1[ix+(iy+1)*nx]+u1[ix+(iy-1)*nx]);
      }
    }

    // update
    for (iy=0; iy<ny; iy++) {
      for (ix=0; ix<nx; ix++) {
        indx = ix+iy*nx;
        u0[indx] = u1[indx];
        u1[indx] = u2[indx];
      }
    }
  }
  stop = clock();
  double cpu_time = (double) (stop-start) / CLOCKS_PER_SEC;
  printf("CPU time = %lf s\n", cpu_time);

  // output the final snapshot
  FILE *file = fopen("u.dat","w");
  fwrite(u2, sizeof(float), nx*ny, file);
  fclose(file);

  // Free memory
  free(u0);
  free(u1);
  free(u2);

  return 0;
}

