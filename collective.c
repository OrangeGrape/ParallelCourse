/* Example of collective communications.
 *
 * This code was written as an example for teaching the course
 * SCPY403/SCPY571 Parallel Programming
 *
 * Copyright 2014, 2018 Chaiwoot Boonyasiriwat. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
int main (int argc, char* argv[]){
  int rank, size, n;
  float result, final;
  MPI_Status status;
  MPI_Request request;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  if (rank == 0) {
    printf("Please enter an integer: ");
    scanf("%d",&n);
  }
  printf("Process %d: n = %d\n", rank, n);

  result = 1.0f;
  MPI_Reduce(&result, &final, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printf("Summation result = %f\n", final);

  MPI_Finalize();
  return 0;
}
