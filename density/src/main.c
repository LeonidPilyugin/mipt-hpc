#include <density.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

#define N_PARTICLES 65536
#define N_FILES 220

#ifndef THREADS
#define THREADS 1
#endif

int main(int argc, char * argv[]) {
    Grid * result[THREADS];
    Grid * temp[THREADS];
    for (i_t i = 0; i < THREADS; i++) {
        result[i] = grid_init(25, 50, -0.5f, 0.5f, -1.5f, 0.5f);
        grid_reset(result[i]);
        temp[i] =grid_init(25, 50, -0.5f, 0.5f, -1.5f, 0.5f);
        grid_reset(temp[i]);
    }
    Particles ** p = malloc(sizeof(Particles *) * N_FILES);
    FILE * fp;
    char filename[4096];

    for (i_t i = 0; i < N_FILES; i++) {
        sprintf(filename, "%s/trajectory%i.csv", argv[1], i);
        fp = fopen(filename, "r");
        p[i] = particles_init_csv(N_PARTICLES, fp);
        fclose(fp);
    }

    DensityParams params;
    params.h = 0.3f;
    params.m = 1.0f;

    double time = omp_get_wtime();

    fflush(stdout);
    #pragma omp parallel num_threads(THREADS)
    {
        i_t thread_id = omp_get_thread_num();
        #pragma omp for
        for (i_t i = 0; i < N_FILES; i++) {
            get_density(p[i], temp[thread_id], &params);
            grid_add(result[thread_id], temp[thread_id]);
        }
    }

    for (i_t i = 1; i < THREADS; i++) {
        grid_add(result[0], result[i]);
    }

    time = omp_get_wtime() - time;
    printf("\nTotal computation time: %lf s\n", time);

    fp = fopen("out.csv", "w");
    grid_write(result[0], fp);
    fclose(fp);

    for (int i = 0; i < N_FILES; i++) {
        particles_destroy(p[i]);
    }
    for (i_t i = 0; i < THREADS; i++) {
        grid_destroy(temp[i]);
        grid_destroy(result[i]);
    }
    free(p);

    return 0;
}
