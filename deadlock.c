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
  int rank, size, n1, n2;
  MPI_Status status;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  // no deadlock
  if (rank == 0) {
    n1 = 10;
    MPI_Send(&n1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&n2, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
    printf( "rank %d: n2 = %d\n", rank, n2 );
  }
  else if (rank == 1) {
    n2 = 20;
    MPI_Send(&n2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&n1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    //MPI_Send(&n2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    printf( "rank %d: n1 = %d\n", rank, n1 );
  }
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
