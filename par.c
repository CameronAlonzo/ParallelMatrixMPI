#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define generate_data(i,j) (i)+(j)*(j)
int main(int argc, char **argv)
{
    int i, j, pid, np, mtag, count;
    //Helper variable for clock and timing performance
    clock_t start, end;
    double t0, t1 ;
    int data[100][100], row_sum[100] ;
    MPI_Status status;
    MPI_Request req_s, req_r;
    // MPI Sets
    MPI_Init( &argc, &argv );
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // generate data[]
    // On  Process 0
    start = clock();
    if(pid == 0) { 
        for(i=0; i<50; i++)
            for(j=0; j<100; j++)
                data[i][j] = generate_data(i,j) ;
        for(i=0; i<50; i++) {
            mtag = 1 ;
            //changed the amount of data being sent to Row by row, as it leads to more overlap
            //Data[i] Should be of size 100, Where Data is of size 10000
            MPI_Isend(data[i], 100, MPI_INT, 1, mtag, MPI_COMM_WORLD, &req_s) ;
            //Have to wait for the send to complete, as we are doing Isend
            MPI_Wait(&req_s, &status) ;
        }
        //Sent the Matrix to Process 1, No change *Necessary* here
        for(i=50; i<100; i++)
            for(j=0; j<100; j++)
                data[i][j] = generate_data(i,j) ;
        for(i=50; i<100; i++) {
            row_sum[i] = 0 ;
            for(j=0; j<100; j++)
                row_sum[i] += data[i][j] ;
        }
        /*** receive computed row_sums from Process 1 ***/
        mtag = 2;
        //Potentially room for further performance in changing this to 100, but unsure.
        MPI_Recv(row_sum, 50, MPI_INT, 1, mtag, MPI_COMM_WORLD, &status) ;
        //Function to print the matrix
        for(i=0; i<100; i++) {
            printf(" %d ", row_sum[i]) ;
            if(i%10 ==9) 
                printf("\n");
            }
        } // End of pid == 0
        else { /*** pid == 1 ***/
            for(i=0; i<50; i++) {
                // You could do this by recieving a block, but it is technically quicker to recieve it row by row. (And makes a bit more sense for scalability reasons)
                mtag = 1 ;
                MPI_Irecv(data[i], 100, MPI_INT, 0, mtag, MPI_COMM_WORLD, &req_r) ;
                //We must wait, as we just did an Irec, leads to message truncation otherwise
                MPI_Wait(&req_r, &status) ;
                //Compute the row sum
                row_sum[i] = 0 ;
                for(j=0; j<100; j++)
                    row_sum[i] += data[i][j] ;
        }
        /*** Send computed row_sums to pid 0 ***/
        mtag = 2 ;
        MPI_Send(row_sum, 50, MPI_INT, 0, mtag, MPI_COMM_WORLD) ;
        }
 MPI_Finalize();
 //Helper code to test to see if the program is indeed running quicker (They both run in 0.000s, this ended up being useless.)
 end = clock();
 double time = (double)(end-start) / CLOCKS_PER_SEC;
 printf("Ran in %f seconds \n", time);
 return 1;
} // End