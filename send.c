/* Example of a point-to-point communication:
 *   Sending/receiving an array and a pointer of size 5.
 *
 * This code was written as an example for teaching the course
 * SCPY403/SCPY571 Parallel Programming
 *
 * Copyright 2014 Chaiwoot Boonyasiriwat. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
int main (int argc, char* argv[]){
  int rank, size, i, num[5], tag;
  float *fnum;
  MPI_Status status;
///////////////////////////////////////////////////////////////////////////////////
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  fnum = (float*) malloc(5*sizeof(float));

  if (rank == 0) {
    tag = 0;
    for (i=0;i<5;i++)
    {
      num[i] = i;
      fnum[i] = (float)i;
    }
    MPI_Send(num, 5, MPI_INT, 1, tag, MPI_COMM_WORLD);
    MPI_Send(fnum, 5, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
  }
  else if (rank == 1) {
    tag = 0;
    MPI_Recv(num, 5, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(fnum, 5, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
//    MPI_Recv(num, 5, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
//    MPI_Recv(fnum, 5, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    for (i=0;i<5;i++) {
      printf( "num[%d] = %d\n", i, num[i] );
      printf( "fnum[%d] = %.1f\n", i, fnum[i] );
    }
  }

  MPI_Finalize();
///////////////////////////////////////////////////////////////////////////////////
  return 0;
}
