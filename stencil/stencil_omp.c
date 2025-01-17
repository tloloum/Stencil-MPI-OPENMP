#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>  /* Pour OpenMP */

#define STENCIL_SIZE 25

typedef float stencil_t;

static const stencil_t alpha = 0.02f;

static const stencil_t epsilon = 0.0001f;

static const int stencil_max_steps = 100000;

static stencil_t *values = NULL;
static stencil_t *prev_values = NULL;

static int size_x = STENCIL_SIZE;
static int size_y = STENCIL_SIZE;

static void stencil_init(void)
{

    values = (stencil_t*) malloc(size_x * size_y * sizeof(stencil_t));
    prev_values = (stencil_t*) malloc(size_x * size_y * sizeof(stencil_t));
    if (!values || !prev_values) {
        fprintf(stderr, "Erreur d’allocation.\n");
        exit(1);
    }

    #pragma omp parallel for schedule(static)
    for(int y = 0; y < size_y; y++)
    {
        for(int x = 0; x < size_x; x++)
        {
            values[x + size_x * y] = 0.0f;
        }
    }

    /* Bord haut et bas */
    #pragma omp parallel for schedule(static)
    for(int x = 0; x < size_x; x++)
    {
        values[x + size_x * 0]            = (stencil_t) x; /* bord du haut */
        values[x + size_x * (size_y - 1)] = (stencil_t)(size_x - x - 1); /* bord du bas */
    }

    /* Bord gauche et droite */
    #pragma omp parallel for schedule(static)
    for(int y = 0; y < size_y; y++)
    {
        values[0 + size_x * y]            = (stencil_t) y; /* bord gauche */
        values[size_x - 1 + size_x * y]   = (stencil_t)(size_y - y - 1); /* bord droit */
    }

    memcpy(prev_values, values, size_x * size_y * sizeof(stencil_t));
}

static void stencil_free(void)
{
    free(values);
    free(prev_values);
}

static void stencil_display(int x0, int x1, int y0, int y1)
{
    for(int y = y0; y <= y1; y++)
    {
        for(int x = x0; x <= x1; x++)
        {
            printf("%8.5g ", values[x + size_x * y]);
        }
        printf("\n");
    }
}

static int stencil_step(void)
{
    stencil_t* tmp = prev_values;
    prev_values = values;
    values = tmp;


    int global_conv = 1;

    #pragma omp parallel
    {
        int local_conv = 1;
        #pragma omp for schedule(static)
        for(int y = 1; y < size_y - 1; y++)
        {
            for(int x = 1; x < size_x - 1; x++)
            {
                /* Calcul de la valeur */
                values[x + size_x * y] =
                    alpha * (
                        prev_values[x - 1 + size_x * y] +
                        prev_values[x + 1 + size_x * y] +
                        prev_values[x + size_x * (y - 1)] +
                        prev_values[x + size_x * (y + 1)]
                    )
                    + (1.0f - 4.0f * alpha) * prev_values[x + size_x * y];

                if(local_conv &&
                   (fabsf(prev_values[x + size_x * y] - values[x + size_x * y]) > epsilon))
                {
                    local_conv = 0;
                }
            }
        }

        #pragma omp atomic
        global_conv &= local_conv;
    }

    return global_conv;
}

int main(int argc, char** argv)
{
    stencil_init();
    printf("# init:\n");
    // stencil_display(0, size_x - 1, 0, size_y - 1);

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    int s;
    for(s = 0; s < stencil_max_steps; s++)
    {
        int convergence = stencil_step();
        if(convergence)
        {
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    const double t_usec = (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_nsec - t1.tv_nsec) / 1.0e3;

    printf("# steps = %d\n", s);
    printf("# time  = %g usecs.\n", t_usec);
    printf("# gflops= %g\n", (6.0 * size_x * size_y * s) / (t_usec * 1000.0));

    // stencil_display(0, size_x - 1, 0, size_y - 1);
    stencil_free();

    return 0;
}

