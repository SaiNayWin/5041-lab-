#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void swap(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

void quicksort(int* arr, int start, int end) {
    if (start >= end) return;
    int pivot = arr[end];
    int i = start - 1;
    for (int j = start; j < end; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr, i, j);
        }
    }
    swap(arr, i + 1, end);
    quicksort(arr, start, i);
    quicksort(arr, i + 2, end);
}

int* merge(int* arr1, int n1, int* arr2, int n2) {
    int* result = (int*)malloc((n1 + n2) * sizeof(int));
    if (!result) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        result[k++] = (arr1[i] < arr2[j]) ? arr1[i++] : arr2[j++];
    }
    while (i < n1) result[k++] = arr1[i++];
    while (j < n2) result[k++] = arr2[j++];
    return result;
}

int main(int argc, char* argv[]) {
    int num_elements, chunk_size, own_chunk_size;
    int *data = NULL, *chunk;
    FILE* file;
    double time_taken;
    MPI_Status status;
    
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        file = fopen(argv[1], "r");
        if (!file) {
            printf("Error opening input file!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fscanf(file, "%d", &num_elements);
        data = (int*)malloc(num_elements * sizeof(int));
        if (!data) {
            printf("Memory allocation failed!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < num_elements; i++) {
            fscanf(file, "%d", &data[i]);
        }
        fclose(file);
    }
    
    MPI_Bcast(&num_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size = (num_elements + num_procs - 1) / num_procs;
    chunk = (int*)malloc(chunk_size * sizeof(int));
    if (!chunk) {
        printf("Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) free(data);
    own_chunk_size = (num_elements >= chunk_size * (rank + 1)) ? chunk_size : (num_elements - chunk_size * rank);
    quicksort(chunk, 0, own_chunk_size - 1);
    
    for (int step = 1; step < num_procs; step *= 2) {
        if (rank % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            free(chunk);
            break;
        }
        if (rank + step < num_procs) {
            int received_size = (num_elements >= chunk_size * (rank + 2 * step)) ? chunk_size * step : (num_elements - chunk_size * (rank + step));
            int* received_chunk = (int*)malloc(received_size * sizeof(int));
            MPI_Recv(received_chunk, received_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);
            data = merge(chunk, own_chunk_size, received_chunk, received_size);
            free(chunk);
            free(received_chunk);
            chunk = data;
            own_chunk_size += received_size;
        }
    }
    
    if (rank == 0) {
        file = fopen(argv[2], "w");
        if (!file) {
            printf("Error opening output file!\n");
            free(chunk);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fprintf(file, "%d\n", own_chunk_size);
        for (int i = 0; i < own_chunk_size; i++) {
            fprintf(file, "%d ", chunk[i]);
        }
        fclose(file);
        free(chunk);
        printf("Sorted output written to %s\n", argv[2]);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}