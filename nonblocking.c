// Nonblocking communication example
//
// Original code from http://beige.ucs.indiana.edu/B673/node153.html
// Modified by Chaiwoot Boonyasiriwat.

#include <stdio.h>
#include <mpi.h>
int main(int argc, char *argv[])
{
  int size, rank, number, count;
  MPI_Request request;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    number = 10;
    MPI_Isend (&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    MPI_Wait (&request, &status);

    if (request == MPI_REQUEST_NULL) {
      printf("\tSend request is completed.\n");
    } else {
      printf("\tSend request is still not completed.\n");
    }
  }
  else if (rank == 1) {
    MPI_Irecv (&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Wait (&request, &status);
    MPI_Get_count (&status, MPI_INT, &count);
    printf("The number of received integer data =  %d\n", count );
    printf("The received number = %d\n", number);

    if (request == MPI_REQUEST_NULL) {
      printf("\tReceive request is completed.\n");
    } else {
      printf("\tReceive request is still not completed.\n");
    }
  }
  MPI_Finalize();
  return 0;
}
