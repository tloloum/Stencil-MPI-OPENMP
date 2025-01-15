#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define STENCIL_SIZE 25
typedef float stencil_t;

/* Paramètres diffusion */
static const stencil_t alpha = 0.02f;
static const stencil_t epsilon = 0.0001f;
/* Nombre max d’itérations */
static const int stencil_max_steps = 100000;

/*
 * Macro d’indexation locale : on stocke un bloc local
 *  de (local_nx+2) * (local_ny+2), y compris les halos.
 */
#define INDEX(sd, x, y) ((y) * ((sd)->local_nx + 2) + (x))

/* Structure globale pour regrouper infos MPI, dimensions, buffers, etc. */
typedef struct {
    /* MPI infos */
    int rank;
    int size;
    MPI_Comm cart_comm;  /* communicateur cartésien */
    int coords[2];       /* coords (px, py) */
    int dims[2];         /* nb de processus dans x et y */

    /* Dimensions globales et locales */
    int global_nx;
    int global_ny;
    int local_nx;
    int local_ny;

    /* Buffers +2 en x,y pour halos */
    stencil_t* values;
    stencil_t* prev_values;

    /* Types dérivés pour échanger les colonnes / lignes */
    MPI_Datatype column_type;
    MPI_Datatype row_type;
} StencilData;

/* =========================================================== */
/* 1) Init MPI, cart comm, etc.                               */
/* =========================================================== */
static void init_mpi(int *argc, char ***argv, StencilData *sd)
{
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &sd->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sd->size);

    sd->global_nx = STENCIL_SIZE;
    sd->global_ny = STENCIL_SIZE;
}

/* =========================================================== */
/* 2) Crée topologie cartésienne 2D                           */
/* =========================================================== */
static void create_cart_2d(StencilData *sd)
{
    /* Factorisation automatique de size en 2D */
    sd->dims[0] = 0;
    sd->dims[1] = 0;
    MPI_Dims_create(sd->size, 2, sd->dims);

    int periods[2] = {0, 0};  /* pas de périodicité */
    MPI_Cart_create(MPI_COMM_WORLD, 2, sd->dims, periods, 0, &sd->cart_comm);

    /* coords cartésiennes dans la grille */
    MPI_Cart_coords(sd->cart_comm, sd->rank, 2, sd->coords);

    /* Taille locale, supposant STENCIL_SIZE divisible par dims */
    sd->local_nx = sd->global_nx / sd->dims[0];
    sd->local_ny = sd->global_ny / sd->dims[1];
}

/* =========================================================== */
/* 3) Allocation des tableaux avec halos                       */
/* =========================================================== */
static void allocate_arrays(StencilData *sd)
{
    int size_with_halo = (sd->local_nx + 2) * (sd->local_ny + 2);
    sd->values = (stencil_t*) malloc(size_with_halo * sizeof(stencil_t));
    sd->prev_values = (stencil_t*) malloc(size_with_halo * sizeof(stencil_t));
    if (!sd->values || !sd->prev_values) {
        fprintf(stderr, "Erreur d’allocation sur le rang %d\n", sd->rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/* =========================================================== */
/* 4) Initialisation des valeurs locales (bord global)         */
/* =========================================================== */
static void init_values(StencilData *sd)
{
    /* Remplir tout à 0 */
    #pragma omp parallel for
    for (int j = 0; j < sd->local_ny + 2; j++) {
        for (int i = 0; i < sd->local_nx + 2; i++) {
            sd->values[INDEX(sd, i, j)] = 0.0f;
            sd->prev_values[INDEX(sd, i, j)] = 0.0f;
        }
    }

    /* Coordonnées globales du point local (i,j): 
       gx = px*local_nx + (i-1)
       gy = py*local_ny + (j-1)
       pour i=1..local_nx, j=1..local_ny (zone intérieure).
    */
    int px = sd->coords[0];
    int py = sd->coords[1];
    int block_nx = sd->local_nx;
    int block_ny = sd->local_ny;

    #pragma omp parallel for
    for (int j = 1; j <= block_ny; j++) {
        for (int i = 1; i <= block_nx; i++) {
            int gx = px * block_nx + (i - 1);
            int gy = py * block_ny + (j - 1);

            /* Bord global en haut/bas/gauche/droite */
            if (gy == 0) {
                sd->values[INDEX(sd, i, j)] = (stencil_t)gx;
            }
            if (gy == sd->global_ny - 1) {
                sd->values[INDEX(sd, i, j)] = (stencil_t)(sd->global_nx - gx - 1);
            }
            if (gx == 0) {
                sd->values[INDEX(sd, i, j)] = (stencil_t)gy;
            }
            if (gx == sd->global_nx - 1) {
                sd->values[INDEX(sd, i, j)] = (stencil_t)(sd->global_ny - gy - 1);
            }

            /* Copie dans prev_values */
            sd->prev_values[INDEX(sd, i, j)] = sd->values[INDEX(sd, i, j)];
        }
    }
}

/* =========================================================== */
/* 5) Création des types dérivés pour colonnes/lignes (MPI)    */
/* =========================================================== */
static void create_mpi_types(StencilData *sd)
{
    /* Type colonne : (local_ny) éléments, stride = (local_nx+2) */
    MPI_Type_vector(sd->local_ny, 1, sd->local_nx + 2, MPI_FLOAT, &sd->column_type);
    MPI_Type_commit(&sd->column_type);

    /* Type ligne : local_nx contigus */
    MPI_Type_contiguous(sd->local_nx, MPI_FLOAT, &sd->row_type);
    MPI_Type_commit(&sd->row_type);
}

/* =========================================================== */
/* 6) Echange des halos (criss-cross)                          */
/* =========================================================== */
static void exchange_halos(StencilData *sd, stencil_t *src)
{
    int rank_left, rank_right;
    int rank_up, rank_down;

    /* dimension=0 => X, dimension=1 => Y */
    MPI_Cart_shift(sd->cart_comm, 0, 1, &rank_left, &rank_right);
    /* On veut rank_up=py-1, rank_down=py+1 => shift en -1 sur la dimension Y */
    MPI_Cart_shift(sd->cart_comm, 1, -1, &rank_up, &rank_down);

    #pragma omp master
    {
        /* ---------------------
         * Echange "haut-bas" 
         *  - 1er appel: envoi top -> up, recv bottom <- down (tag=101)
         *  - 2e appel: envoi bottom -> down, recv top <- up (tag=102)
         * --------------------- */
        /* Call 1 */
        MPI_Sendrecv(
            /* top row = j=1 */
            &src[INDEX(sd, 1, 1)], 1, sd->row_type,  /* send */
            rank_up, 101,
            &src[INDEX(sd, 1, sd->local_ny+1)], 1, sd->row_type,  /* recv */
            rank_down, 101,
            sd->cart_comm, MPI_STATUS_IGNORE
        );
        /* Call 2 */
        MPI_Sendrecv(
            /* bottom row = j=local_ny */
            &src[INDEX(sd, 1, sd->local_ny)], 1, sd->row_type,  
            rank_down, 102,
            &src[INDEX(sd, 1, 0)], 1, sd->row_type,  
            rank_up, 102,
            sd->cart_comm, MPI_STATUS_IGNORE
        );

        /* ---------------------
         * Echange "gauche-droite"
         *  - 3e appel: envoi left -> rank_left, recv right <- rank_right (tag=201)
         *  - 4e appel: envoi right -> rank_right, recv left <- rank_left (tag=202)
         * --------------------- */
        /* Call 3 */
        MPI_Sendrecv(
            &src[INDEX(sd, 1, 1)], 1, sd->column_type,
            rank_left, 201,
            &src[INDEX(sd, sd->local_nx+1, 1)], 1, sd->column_type,
            rank_right, 201,
            sd->cart_comm, MPI_STATUS_IGNORE
        );
        /* Call 4 */
        MPI_Sendrecv(
            &src[INDEX(sd, sd->local_nx, 1)], 1, sd->column_type,
            rank_right, 202,
            &src[INDEX(sd, 0, 1)], 1, sd->column_type,
            rank_left, 202,
            sd->cart_comm, MPI_STATUS_IGNORE
        );
    }
    /* Barrière de synchronisation entre threads */
    #pragma omp barrier
}

/* =========================================================== */
/* 7) Calcul d’un pas : swap, échange halos, mise à jour       */
/*    Retourne 1 si globalement convergent, 0 sinon            */
/* =========================================================== */
static int do_step(StencilData *sd)
{
    /* Swap buffers */
    stencil_t* tmp = sd->prev_values;
    sd->prev_values = sd->values;
    sd->values = tmp;

    /* Echange des halos (un seul thread fait MPI, tous attendent) */
    exchange_halos(sd, sd->prev_values);

    /* Calcul + test de convergence local */
    int global_conv = 1;
    #pragma omp parallel
    {
        int local_conv = 1;
        #pragma omp for
        for (int j = 1; j <= sd->local_ny; j++) {
            for (int i = 1; i <= sd->local_nx; i++) {
                sd->values[INDEX(sd, i, j)] =
                    alpha * (
                        sd->prev_values[INDEX(sd, i-1, j)] +
                        sd->prev_values[INDEX(sd, i+1, j)] +
                        sd->prev_values[INDEX(sd, i, j-1)] +
                        sd->prev_values[INDEX(sd, i, j+1)]
                    )
                    + (1.0f - 4.0f*alpha) * sd->prev_values[INDEX(sd, i, j)];

                if (local_conv &&
                    (fabsf(sd->values[INDEX(sd, i, j)]
                          - sd->prev_values[INDEX(sd, i, j)]) > epsilon)) {
                    local_conv = 0;
                }
            }
        }
        #pragma omp atomic
        global_conv &= local_conv;
    }

    /* Réduction MPI (logique AND) pour tester la convergence globale */
    int global_result = 0;
    MPI_Allreduce(&global_conv, &global_result, 1, MPI_INT, MPI_LAND, sd->cart_comm);

    return global_result;
}

/* =========================================================== */
/* 8) (Optionnel) affichage global sur le rang 0               */
/* =========================================================== */
static void print_global_array(StencilData *sd)
{
    int px = sd->coords[0];
    int py = sd->coords[1];
    int block_nx = sd->local_nx;
    int block_ny = sd->local_ny;
    int global_nx = sd->global_nx;
    int global_ny = sd->global_ny;

    int local_size = block_nx * block_ny;
    stencil_t* local_buf = (stencil_t*) malloc(local_size * sizeof(stencil_t));
    if (!local_buf) return;

    /* Copie locale (sans halo) */
    #pragma omp parallel for
    for (int j = 0; j < block_ny; j++) {
        for (int i = 0; i < block_nx; i++) {
            local_buf[j*block_nx + i] = sd->values[INDEX(sd, i+1, j+1)];
        }
    }

    /* Rassemblement sur le rang 0 */
    stencil_t* global_buf = NULL;
    if (sd->rank == 0) {
        global_buf = (stencil_t*) malloc(global_nx * global_ny * sizeof(stencil_t));
    }

    if (sd->rank == 0) {
        /* Copie locale d’abord (moi-même) */
        int gox = px * block_nx;
        int goy = py * block_ny;
        for (int j = 0; j < block_ny; j++) {
            memcpy(&global_buf[(goy + j)*global_nx + gox],
                   &local_buf[j*block_nx],
                   block_nx * sizeof(stencil_t));
        }

        /* Réception depuis les autres ranks */
        MPI_Status st;
        for (int r = 1; r < sd->size; r++) {
            int coords_sender[2];
            MPI_Recv(coords_sender, 2, MPI_INT, r, 999, sd->cart_comm, &st);
            int sx = coords_sender[0];
            int sy = coords_sender[1];
            int gox2 = sx * block_nx;
            int goy2 = sy * block_ny;
            MPI_Recv(local_buf, local_size, MPI_FLOAT, r, 998, sd->cart_comm, &st);
            for (int j = 0; j < block_ny; j++) {
                memcpy(&global_buf[(goy2 + j)*global_nx + gox2],
                       &local_buf[j*block_nx],
                       block_nx * sizeof(stencil_t));
            }
        }

        /* Affichage final */
        for (int j = 0; j < global_ny; j++) {
            for (int i = 0; i < global_nx; i++) {
                printf("%8.5g ", global_buf[i + j*global_nx]);
            }
            printf("\n");
        }
        free(global_buf);
    } else {
        int coords_buf[2] = {px, py};
        MPI_Send(coords_buf, 2, MPI_INT, 0, 999, sd->cart_comm);
        MPI_Send(local_buf, local_size, MPI_FLOAT, 0, 998, sd->cart_comm);
    }

    free(local_buf);
}

/* =========================================================== */
/* 9) Libération et finalisation MPI                           */
/* =========================================================== */
static void clean_up(StencilData *sd)
{
    free(sd->values);
    free(sd->prev_values);
    MPI_Type_free(&sd->column_type);
    MPI_Type_free(&sd->row_type);
    MPI_Comm_free(&sd->cart_comm);
    MPI_Finalize();
}

/* =========================================================== */
/* Programme principal                                        */
/* =========================================================== */
int main(int argc, char** argv)
{
    StencilData sd;
    init_mpi(&argc, &argv, &sd);
    create_cart_2d(&sd);
    allocate_arrays(&sd);
    init_values(&sd);
    create_mpi_types(&sd);

    /* Mesure du temps */
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    int s;
    int converged_global = 0; /* 1 => convergé */
    for (s = 0; s < stencil_max_steps; s++) {
        int is_conv = do_step(&sd);
        if (is_conv) {
            converged_global = 1;
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed_usec = (t2.tv_sec - t1.tv_sec)*1e6 + (t2.tv_nsec - t1.tv_nsec)/1e3;

    /* Affichage perf sur le rang 0 */
    if (sd.rank == 0) {
        printf("# steps = %d\n", s);
        printf("# time  = %g usecs\n", elapsed_usec);
        double flop = 6.0 * sd.global_nx * sd.global_ny * s;  /* 6 ops par point/itération */
        double gflops = flop / (elapsed_usec * 1000.0);
        printf("# gflops= %g\n", gflops);
    }

    /* (Optionnel) afficher le résultat final */
    // if (sd.rank == 0) printf("=== Résultat final ===\n");
    // print_global_array(&sd);

    clean_up(&sd);
    return 0;
}

