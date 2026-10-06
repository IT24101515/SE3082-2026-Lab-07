#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    // Only Rank 0 stores the full array
    int *array = NULL;

    if (rank == 0)
    {
        array = (int *)malloc(N * sizeof(int));

        for (int i = 0; i < N; i++)
        {
            array[i] = i + 1;
        }

        printf("Root filled array with values 1 to %d\n", N);
    }

    // Each rank stores only its own chunk
    int *local_chunk =
        (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    // Divide the original array among all ranks
    MPI_Scatter(
        array,
        chunk_size,
        MPI_INT,
        local_chunk,
        chunk_size,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Each rank calculates its local sum
    long long local_sum = 0;

    for (int i = 0; i < chunk_size; i++)
    {
        local_sum += local_chunk[i];
    }

    printf("Rank %d: local_sum = %lld\n",
           rank, local_sum);

    // Only Rank 0 needs space to collect all local sums
    long long *all_sums = NULL;

    if (rank == 0)
    {
        all_sums =
            (long long *)malloc(size * sizeof(long long));
    }

    // Gather one local_sum from every rank into Rank 0
    MPI_Gather(
        &local_sum,
        1,
        MPI_LONG_LONG,
        all_sums,
        1,
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    // Rank 0 adds all gathered local sums
    if (rank == 0)
    {
        long long total_sum = 0;

        for (int i = 0; i < size; i++)
        {
            total_sum += all_sums[i];
        }

        double elapsed = MPI_Wtime() - start;

        long long expected =
            (long long)N * (N + 1) / 2;

        printf("\n[Gather] Total sum = %lld\n",
               total_sum);

        printf("[Gather] Expected  = %lld\n",
               expected);

        printf("[Gather] Correct?  = %s\n",
               total_sum == expected ? "YES" : "NO");

        printf("[Gather] Time      = %.4f sec\n",
               elapsed);

        free(all_sums);
    }

    free(local_chunk);

    if (rank == 0)
    {
        free(array);
    }

    MPI_Finalize();

    return 0;
}
