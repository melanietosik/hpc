#include <math.h>
#include <stdio.h> 
#include <time.h> 

// 5GB
#define ARRAY_SIZE (5 * (int)(1e9 / 8))


int main () {

    struct timespec start, end;

    int i;
    int n;
    double array[ARRAY_SIZE];
    double sum;

    // printf("[debug] ARRAY_SIZE: %d\n", ARRAY_SIZE);
    // printf("[debug] sizeof(array): %lu\n", sizeof(array));

    double lowest_c1 = INFINITY;
    double lowest_c2 = INFINITY;

    /*
        C1
    */
    for (n = 0; n < 10; n++) {

        // Initialization
        sum = 0;
        for (i = 0; i < ARRAY_SIZE; ++i)
            array[i] = (double)i/3;

        clock_gettime(CLOCK_MONOTONIC, &start);  // Start
        
        for (i = 0; i < ARRAY_SIZE; i++) {
            sum += array[i]*2;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);    // End

        double time_sec =
            ((double)end.tv_sec + (double)end.tv_nsec / 1e9) - 
            ((double)start.tv_sec + (double)start.tv_nsec / 1e9);

        if (time_sec < lowest_c1)
            lowest_c1 = time_sec;

        printf("[debug] Sum n=%d: %.02lf\n", n, sum);
    }
    printf("[output] *C1*\n");
    printf("[output] Time: %.02lf secs\n", lowest_c1);
    printf("[output] Bandwidth: %.02lf GB/s\n", (sizeof(array) / lowest_c1) / 1e9);
    printf("[output] FLOPS: %.02lf GFLOP/s\n\n", ((2 * ARRAY_SIZE / lowest_c1) / 1e9));

    /*
        C2
    */
    for (n = 0; n < 10; n++) {
        
        // Initialization
        sum = 0;
        for (i = 0; i < ARRAY_SIZE; ++i)
            array[i] = (double)i/3;
        
        clock_gettime(CLOCK_MONOTONIC,&start);  // Start
        
        for (i = 0; i < ARRAY_SIZE; i+=8) {
            sum += array[i]*2;
            sum += array[i+1]*2;
            sum += array[i+2]*2;
            sum += array[i+3]*2;
            sum += array[i+4]*2;
            sum += array[i+5]*2;
            sum += array[i+6]*2;
            sum += array[i+7]*2;
        }
        
        clock_gettime(CLOCK_MONOTONIC,&end);    // End

        double time_sec =
            ((double)end.tv_sec + (double)end.tv_nsec / 1e9) - 
            ((double)start.tv_sec + (double)start.tv_nsec / 1e9);

        if (time_sec < lowest_c2)
            lowest_c2 = time_sec;

        printf("[debug] Sum n=%d: %.02lf\n", n, sum);
    }
    printf("[output] *C2*\n");
    printf("[output] Time: %.02lf secs\n", lowest_c2);
    printf("[output] Bandwidth: %.02lf GB/s\n", (sizeof(array) / lowest_c2) / 1e9);
    printf("[output] FLOPS: %.02lf GFLOP/s\n", ((2 * ARRAY_SIZE / lowest_c2) / 1e9));

    return(0);
}
