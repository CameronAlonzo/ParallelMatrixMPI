#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>//so we can use time for random seed

#define INFINITY 99999999 // Define a large number to represent infinity.
#define min(x, y) (((x) < (y)) ? (x) : (y)) // Macro to find the minimum of two values.

// Function to generate a random matrix with zeros on the diagonal.
void generateRandomMatrix(int n, int *matrix) {
    srand(time(NULL));//using current time as seed
    int i, j; // Loop counters declared outside the loops for compatibility with older C standards.
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix[i * n + j] = (i == j) ? 0 : (rand() % 100 + 1); // Assign 0 to diagonal elements, random values to others.
        }
    }
}

// Function to print the matrix.
void printMatrix(int n, int *matrix) {
    int i, j; // Loop counters.
    printf("Matrix:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%4d ", matrix[i * n + j]); // Print each element with padding for alignment.
        }
        printf("\n");
    }
}

// The parallel algorithm to solve the problem using MPI.
void parallelHW2(int n, int *matrix, int *output, int rank, int size) {
    int *done = (int *)calloc(n, sizeof(int)); // Array to keep track of processed nodes.
    int count = 1, i; // Count initialized to 1 to account for the starting node.

    // Initialization by the root process.
    if (rank == 0) {
        for (i = 0; i < n; i++) {
            output[i] = matrix[i]; // Copy the first row to the output array.
            done[i] = 0; // Initialize all nodes as not done.
        }
        done[0] = 1; // Mark the first node as done.
    }

    // Broadcast the initialized output and done arrays to all processes.
    MPI_Bcast(output, n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(done, n, MPI_INT, 0, MPI_COMM_WORLD);

    // Main loop to update the output array until all nodes are processed.
    while (count < n) {
        int localLeastVal = INFINITY, localLeastPos = -1, globalLeastVal, globalLeastPos;

        // Each process finds its local minimum.
        for (i = 0; i < n; i++) {
            if (!done[i] && output[i] < localLeastVal) {
                localLeastVal = output[i];
                localLeastPos = i;
            }
        }

        // Reduce operation to find the global minimum value and its position.
        MPI_Allreduce(&localLeastVal, &globalLeastVal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Determine if the local minimum is also the global minimum.
        int isLocalMinimum = (localLeastVal == globalLeastVal) ? 1 : 0;
        globalLeastPos = -1;

        // Only the process that found the local minimum updates the done array.
        if (isLocalMinimum) {
            done[localLeastPos] = 1;
            globalLeastPos = localLeastPos;
        }

        // Broadcast the position of the global minimum to all processes.
        MPI_Bcast(&globalLeastPos, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Update the output array based on the global least value.
        for (i = 0; i < n; i++) {
            if (!done[i]) {
                output[i] = min(output[i], globalLeastVal + matrix[globalLeastPos * n + i]);
            }
        }

        // Synchronize the done array and increment the count.
        MPI_Bcast(done, n, MPI_INT, 0, MPI_COMM_WORLD);
        count++;
        MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(done); // Free the allocated memory for the done array.
}

// The main function initializes MPI, calls the parallel algorithm, and cleans up.
int main(int argc, char *argv[]) {
    int rank, size, n = 8, i;
    MPI_Init(&argc, &argv); // Initialize MPI environment.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process.
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes.

    int *matrix = (int *)malloc(n * n * sizeof(int)); // Allocate memory for the matrix
    int *output = (int *)malloc(n * sizeof(int)); // Allocate memory for the output array.

    // Generate and print the matrix on the root process.
    if (rank == 0) {
        generateRandomMatrix(n, matrix);
        printMatrix(n, matrix);
    }

    // Broadcast the matrix to all processes.
    MPI_Bcast(matrix, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // Call the parallel algorithm.
    parallelHW2(n, matrix, output, rank, size);

    // Print the final output array on the root process.
    if (rank == 0) {
        printf("Final output array:\n");
        for (i = 0; i < n; i++) {
            printf("%d ", output[i]);
        }
        printf("\n");
    }

    // Free allocated memory and finalize the MPI environment.
    free(matrix);
    free(output);

    MPI_Finalize();
    return 0;
}






