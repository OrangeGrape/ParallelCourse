#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

typedef struct
{
  float x;  // real part
  float y;  // imag part
} Complex;


void main(int argc, char** argv) {
  int max_iter = 10000;
  int nx = 600;
  int ny = 600;
  int i, j, k, nxi, nxf;
  int data[nx][ny];
  Complex z, c;
  float lengthsq, temp;
  FILE *file;

  int rank, n, source;

/////////////////////////////////////////////////////////////////

  MPI_Request request;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  
  nxi= rank*(nx/n +1);
  nxf= (rank+1)*(nx/n +1);

  if (nxf>nx){
    nxf = nx;
  }

  for(i=nxi; i < nxf; i++) {
    for(j=0; j < ny; j++) {
      z.x = z.y = 0.0;
      c.x = (4.0*j - 2.0*nx)/nx;
      c.y = (4.0*i - 2.0*ny)/ny;
      k = 0;
      do {
        temp = z.x*z.x - z.y*z.y + c.x;
        z.y = 2.0*z.x*z.y + c.y;
        z.x = temp;
        lengthsq = z.x*z.x+z.y*z.y;
        k++;
      } while (lengthsq < 4.0 && k < max_iter);
      if (k == max_iter)
        k = 0;
      else
          k = k%16;
      data[i][j] = k;
    }
  }

//send result to rank 0
  if(rank != 0){

    for(i=nxi; i < nxf; i++) {
      for(j=0; j < ny; j++) {
        MPI_Send (&data[i][j], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      }
    }

  }else if (rank == 0){

    for(i= nx/n +1; i < nx; i++) {
      for(j=0; j < ny; j++) {
        source = i/(nx/n +1);
        MPI_Recv (&data[i][j], 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &status);
      }
    }
  }  
//save data
  if(rank == 0){
    file = fopen("MPImandel.ppm","w");
    fprintf(file, "P2 %d %d 16\n", nx, ny);
    for(i=0; i < nx; i++) {
      for(j=0; j < ny; j++) {
        fprintf(file, "%d \t", data[i][j]);
      }
      fprintf(file, "\n");
    }
    fprintf(file, "\n");
    fclose(file);    
  }
  
  MPI_Finalize();
  
}
