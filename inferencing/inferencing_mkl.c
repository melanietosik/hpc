#include <math.h>
#include <mkl.h>
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>

#define I_0 (256*256)    // Input [l=0]
#define H_0 4000         // Hidden layer [l=0]
#define H_1 1000         // Hidden layer [l=1]


double* init(int row, int col) {
    /*
        Initialization
    */
    // Matrix is represented as array for simplicity
    double* matrix = calloc(row * col, sizeof(double));
    int i, j;

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix[i * col + j] = (0.4 + ((i + j) % 40 - 20) / 40.0);
        }
    }
    return matrix;
}


double* multiply(double* A, double* B, int a_row, int a_col, int b_col) {
    /*
        Matrix multiplication
    */
    double* result = (double*)calloc(a_row * b_col, sizeof(double));

    int i, j, k;
    for (i = 0; i < a_row; i++) {
        for (j = 0; j < b_col; j++) {
            for (k = 0; k < a_col; k++) {
                result[i * b_col + j] += A[i * a_col + k] * B[k * b_col + j];
            }
        }
    }
    return result;
}


double* multiply_mkl(double* A, double* B, int a_row, int a_col, int b_col) {
    /*
        Matrix multiplication using Intel MKL
    */
    double* result = (double*)calloc(a_row * b_col, sizeof(double));

    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        (MKL_INT)a_row,
        (MKL_INT)a_col,
        1,
        A,
        a_col,
        B,
        1,
        0,
        result,
        1
    );
    return result;
}


double* relu(double* matrix, int w, int h) {
    /*
        ReLU (rectified linear unit) activation function
    */
    int i;
    for (i = 0; i < w * h; i++) {
        if (matrix[i] < 0)
            matrix[i] = 0;
    }
    return matrix;
}


void c5() {
    /*
        C5
    */
    printf("*C5*\n");
    struct timespec start, end;

    double* x_0 = init(I_0, 1);
    double* W_0 = init(H_0, I_0);
    double* W_1 = init(H_1, H_0);

    /* Q5 */
    // printf("Size of x_0 in memory: %lu\n", (I_0 * 1 * sizeof(double)));
    // printf("Size of W_0 in memory: %lu\n", (H_0 * I_0 * sizeof(double)));
    // printf("Size of W_1 in memory: %lu\n", (H_1 * H_0 * sizeof(double)));

    clock_gettime(CLOCK_MONOTONIC, &start);  // Start
    double* z_0 = relu(multiply(W_0, x_0, H_0, I_0, 1), H_0, 1);
    double* z_1 = relu(multiply(W_1, z_0, H_1, H_0, 1), H_1, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);    // End

    /* Q5 */
    // printf("Size of z_0 in memory: %lu\n", (H_0 * 1 * sizeof(double)));
    // printf("Size of z_1 in memory: %lu\n", (H_1 * 1 * sizeof(double)));

    // Compute execution time
    double exec_time =
        ((double)end.tv_sec + (double)end.tv_nsec / 1e9) - 
        ((double)start.tv_sec + (double)start.tv_nsec / 1e9);
    printf("exectime=%.02lf\n", exec_time);

    // Compute checksum
    double checksum = 0.0;
    int i, j;
    for (i = 0; i < H_1; i++) {
        for (j = 0; j < 1; j++) {
            checksum += z_1[i + j];
        }
    }
    printf("checksum=%0.2lf\n", checksum);

    // Free up memory
    free(x_0);
    free(W_0);
    free(W_1);
    free(z_0);
    free(z_1);

    return;
}


void c6() {
    /*
        C6
    */
    printf("*C6*\n");
    struct timespec start, end;

    double* x_0 = init(I_0, 1);
    double* W_0 = init(H_0, I_0);
    double* W_1 = init(H_1, H_0);

    clock_gettime(CLOCK_MONOTONIC, &start);  // Start
    double* z_0 = relu(multiply_mkl(W_0, x_0, H_0, I_0, 1), H_0, 1);
    double* z_1 = relu(multiply_mkl(W_1, z_0, H_1, H_0, 1), H_1, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);    // End

    // Compute execution time
    double exec_time =
        ((double)end.tv_sec + (double)end.tv_nsec / 1e9) - 
        ((double)start.tv_sec + (double)start.tv_nsec / 1e9);
    printf("exectime=%.02lf\n", exec_time);

    // Compute checksum
    double checksum = 0.0;
    int i, j;
    for (i = 0; i < H_1; i++) {
        for (j = 0; j < 1; j++) {
            checksum += z_1[i + j];
        }
    }
    printf("checksum=%0.2lf\n", checksum);

    // Free up memory
    free(x_0);
    free(W_0);
    free(W_1);
    free(z_0);
    free(z_1);

    return;
}


int main() {
    c5();
    c6();
    return 0;
}
