#include <stdio.h>
#include <mpi.h>
void main(int argc, char** argv) {
  int rank, n;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  if (rank == 0)
    printf("number of processes = %d\n", n);
  printf("Hello from rank %d\n", rank);
  MPI_Finalize();
}

