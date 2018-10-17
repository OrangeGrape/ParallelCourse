#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


void main(int argc, char** argv) {
  int nx = 1000;
  int i, nxi, nxf;
  float xi, xmin = -10, xmax = 10, dx = (xmax-xmin)/nx;
  float fx[nx], sum,tmp;
  //initial
  
  for(i=0;i<nx;i++){
    xi = xmin + i*dx;
    fx[i] = (xi*xi) + (2*xi) + 1;
  }
  
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
  
  sum = 0;
  for(i=nxi; i < nxf-1; i++) {
    sum += (fx[i]+fx[i+1])*(dx/2);
  }

//send result to rank 0
  if(rank != 0){
    MPI_Send (&sum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }else if (rank == 0){
    for(source = 1; source < n; source++){
      MPI_Recv (&tmp, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &status);
      sum+=tmp;
    }
  }  
//save data
  if(rank == 0){
    printf("Integrate result is: %f \n", sum);  
  }
  
  MPI_Finalize();
  
}
