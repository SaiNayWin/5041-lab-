#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Swap function
void swap(int* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

// Corrected QuickSort function
void quicksort(int* arr, int left, int right) {
    if (left >= right) return;

    int pivot = arr[right];
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr, i, j);
        }
    }

    swap(arr, i + 1, right);
    int partitionIndex = i + 1;

    quicksort(arr, left, partitionIndex - 1);
    quicksort(arr, partitionIndex + 1, right);
}

// Merge function
int* merge(int* arr1, int n1, int* arr2, int n2) {
    int* result = (int*)malloc((n1 + n2) * sizeof(int));
    int i = 0, j = 0, k = 0;

    while (i < n1 && j < n2) {
        if (arr1[i] < arr2[j])
            result[k++] = arr1[i++];
        else
            result[k++] = arr2[j++];
    }
    while (i < n1) result[k++] = arr1[i++];
    while (j < n2) result[k++] = arr2[j++];

    return result;
}

int main(int argc, char* argv[]) {
    int num_elements, *data = NULL;
    int chunk_size, own_chunk_size, *chunk;
    FILE* file;
    double start_time;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0)
            printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (rank == 0) {
        printf("[Process %d] Reading input file: %s\n", rank, argv[1]);
        file = fopen(argv[1], "r");
        if (!file) {
            printf("[Process %d] Error opening input file.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fscanf(file, "%d", &num_elements);
        printf("[Process %d] Number of elements: %d\n", rank, num_elements);

        data = (int*)malloc(num_elements * sizeof(int));

        printf("[Process %d] Reading array from file...\n", rank);
        for (int i = 0; i < num_elements; i++) {
            fscanf(file, "%d", &data[i]);
        }
        fclose(file);

        printf("[Process %d] Original array: ", rank);
        for (int i = 0; i < num_elements; i++)
            printf("%d ", data[i]);
        printf("\n");
    }

    MPI_Bcast(&num_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size = (num_elements + num_procs - 1) / num_procs;

    chunk = (int*)malloc(chunk_size * sizeof(int));
    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) free(data);

    own_chunk_size = (num_elements >= chunk_size * (rank + 1))
                     ? chunk_size
                     : (num_elements - chunk_size * rank);

    printf("[Process %d] Received %d elements for sorting.\n", rank, own_chunk_size);

    quicksort(chunk, 0, own_chunk_size - 1);

    printf("[Process %d] Sorted chunk: ", rank);
    for (int i = 0; i < own_chunk_size; i++)
        printf("%d ", chunk[i]);
    printf("\n");

    for (int step = 1; step < num_procs; step *= 2) {
        if (rank % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            printf("[Process %d] Sent sorted chunk to process %d\n", rank, rank - step);
            break;
        }

        if (rank + step < num_procs) {
            int recv_size = (num_elements >= chunk_size * (rank + 2 * step))
                            ? chunk_size * step
                            : (num_elements - chunk_size * (rank + step));

            int* recv_chunk = (int*)malloc(recv_size * sizeof(int));
            MPI_Recv(recv_chunk, recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);
            printf("[Process %d] Received sorted chunk from process %d\n", rank, rank + step);

            int* merged_chunk = merge(chunk, own_chunk_size, recv_chunk, recv_size);
            free(chunk);
            free(recv_chunk);
            chunk = merged_chunk;
            own_chunk_size += recv_size;
        }
    }

    if (rank == 0) {
        printf("\n[Process %d] Final sorted array: ", rank);
        for (int i = 0; i < own_chunk_size; i++)
            printf("%d ", chunk[i]);
        printf("\n");

        file = fopen(argv[2], "w");
        if (!file) {
            printf("[Process %d] Error opening output file.\n", rank);
            exit(EXIT_FAILURE);
        }

        fprintf(file, "Sorted array:\n");
        for (int i = 0; i < own_chunk_size; i++) {
            fprintf(file, "%d ", chunk[i]);
        }
        fclose(file);

        printf("[Process %d] Sorted array written to %s\n", rank, argv[2]);
    }

    free(chunk);
    MPI_Finalize();
    return 0;
}
