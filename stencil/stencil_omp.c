#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>  /* Pour OpenMP */

#define STENCIL_SIZE 25

typedef float stencil_t;

/** conduction coeff used in computation */
static const stencil_t alpha = 0.02f;

/** threshold for convergence */
static const stencil_t epsilon = 0.0001f;

/** max number of steps */
static const int stencil_max_steps = 100000;

static stencil_t *values = NULL;
static stencil_t *prev_values = NULL;

static int size_x = STENCIL_SIZE;
static int size_y = STENCIL_SIZE;

/** init stencil values to 0, borders to non-zero */
static void stencil_init(void)
{
    /* 
     * On peut éventuellement utiliser _aligned_malloc sous Windows 
     * ou posix_memalign sous Linux pour aligner la mémoire si on veut plus d'optimisations SIMD,
     * mais pour STENCIL_SIZE=25, c'est moins critique.
     */
    values = (stencil_t*) malloc(size_x * size_y * sizeof(stencil_t));
    prev_values = (stencil_t*) malloc(size_x * size_y * sizeof(stencil_t));
    if (!values || !prev_values) {
        fprintf(stderr, "Erreur d’allocation.\n");
        exit(1);
    }

    /* Initialisation à zéro (ou presque) en parallèle */
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

    /* Copie dans prev_values */
    memcpy(prev_values, values, size_x * size_y * sizeof(stencil_t));
}

static void stencil_free(void)
{
    free(values);
    free(prev_values);
}

/** display a (part of) the stencil values */
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

/** compute the next stencil step, return 1 if computation has converged */
static int stencil_step(void)
{
    /* switch buffers */
    stencil_t* tmp = prev_values;
    prev_values = values;
    values = tmp;

    /* 
     * On fait le calcul au coeur du stencil (1..size_x-2, 1..size_y-2).
     * On utilise une variable locale 'local_conv' pour mesurer la convergence
     * en parallèle, puis on fait une réduction logique AND (&&).
     */
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

                /* Test de convergence locale */
                if(local_conv &&
                   (fabsf(prev_values[x + size_x * y] - values[x + size_x * y]) > epsilon))
                {
                    local_conv = 0;
                }
            }
        }

        /* Réduction manuelle du local_conv dans global_conv */
        #pragma omp atomic
        global_conv &= local_conv;
    }

    return global_conv;
}

int main(int argc, char** argv)
{
    /* 
     * On peut éventuellement forcer un nombre de threads en C:
     *   omp_set_num_threads(4);
     * sinon on laisse l'utilisateur gérer (ex: OMP_NUM_THREADS=4 ./stencil_omp).
     */

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

