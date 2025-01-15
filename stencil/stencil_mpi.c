#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define STENCIL_SIZE 25
typedef float stencil_t;

/** Coefficient et seuil pour la diffusion */
static const stencil_t alpha = 0.02;
static const stencil_t epsilon = 0.0001;
/** Nombre max d’itérations */
static const int stencil_max_steps = 100000;

/* Macro pour indexer (x, y) dans un tableau local (avec halo) de dimension (local_nx+2) * (local_ny+2). */
#define IDX(sd, x, y) ((y) * ((sd)->local_nx + 2) + (x))

/* Structure contenant tout ce dont on a besoin. */
typedef struct {
    /* MPI */
    int rank;
    int size;
    MPI_Comm cart_comm;
    int coords[2];
    int dims[2];

    /* Dimensions globales et locales (sans halo). */
    int global_nx;
    int global_ny;
    int local_nx;
    int local_ny;

    /* Buffers : on prend +2 en x et y pour les halos. */
    stencil_t* values;
    stencil_t* prev_values;

    /* Types dérivés pour l’échange de halos (lignes et colonnes). */
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
/* 2) Créer topologie cartésienne 2D                          */
/* =========================================================== */
static void create_cart_2d(StencilData *sd)
{
    /* Dimensions automatiques avec MPI_Dims_create. */
    sd->dims[0] = 0;
    sd->dims[1] = 0;
    MPI_Dims_create(sd->size, 2, sd->dims);

    int periods[2] = {0, 0};  /* pas de périodicité */
    MPI_Cart_create(MPI_COMM_WORLD, 2, sd->dims, periods, 0, &sd->cart_comm);

    /* Récupère les coordonnées cartésiennes (px, py). */
    MPI_Cart_coords(sd->cart_comm, sd->rank, 2, sd->coords);

    /* Calcul de la taille locale (sans halo). */
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
/* 4) Initialisation des valeurs locales                       */
/*    en tenant compte des conditions de bord globales         */
/* =========================================================== */
static void init_values(StencilData *sd)
{
    /* On met tout à 0 au départ. */
    for (int j = 0; j < sd->local_ny + 2; j++) {
        for (int i = 0; i < sd->local_nx + 2; i++) {
            sd->values[IDX(sd, i, j)] = 0.0f;
            sd->prev_values[IDX(sd, i, j)] = 0.0f;
        }
    }

    /* On remplit les bords (si on est sur un bord global). 
       Les coordonnées globales d’un point local (i, j) sont :
         gx = px*local_nx + (i - 1)
         gy = py*local_ny + (j - 1)
       pour i=1..local_nx, j=1..local_ny (la zone intérieure).
    */
    int px = sd->coords[0];
    int py = sd->coords[1];

    for (int j = 1; j <= sd->local_ny; j++) {
        for (int i = 1; i <= sd->local_nx; i++) {
            int gx = px * sd->local_nx + (i - 1);
            int gy = py * sd->local_ny + (j - 1);

            // Bord du haut : y=0 => value = x
            if (gy == 0) {
                sd->values[IDX(sd, i, j)] = (stencil_t)gx;
            }
            // Bord du bas : y=STENCIL_SIZE-1 => value = size_x - x - 1
            if (gy == sd->global_ny - 1) {
                sd->values[IDX(sd, i, j)] = (stencil_t)(sd->global_nx - gx - 1);
            }
            // Bord de gauche : x=0 => value = y
            if (gx == 0) {
                sd->values[IDX(sd, i, j)] = (stencil_t)gy;
            }
            // Bord de droite : x=STENCIL_SIZE-1 => value = size_y - y - 1
            if (gx == sd->global_nx - 1) {
                sd->values[IDX(sd, i, j)] = (stencil_t)(sd->global_ny - gy - 1);
            }

            sd->prev_values[IDX(sd, i, j)] = sd->values[IDX(sd, i, j)];
        }
    }
}

/* =========================================================== */
/* 5) Création des types dérivés MPI pour colonnes/lignes      */
/* =========================================================== */
static void create_mpi_types(StencilData *sd)
{
    /* Type colonne : block_ny éléments, stride=(local_nx+2). */
    MPI_Type_vector(sd->local_ny, 1, sd->local_nx + 2, MPI_FLOAT, &sd->column_type);
    MPI_Type_commit(&sd->column_type);

    /* Type ligne : block_nx contigus. */
    MPI_Type_contiguous(sd->local_nx, MPI_FLOAT, &sd->row_type);
    MPI_Type_commit(&sd->row_type);
}

/* =========================================================== */
/* 6) Echange des halos : on utilise MPI_Sendrecv              */
/* =========================================================== */
static void exchange_halos(StencilData *sd, stencil_t *src)
{
    int rank_left, rank_right;
    int rank_up, rank_down;

    // dimension=0 => X, dimension=1 => Y
    // On veut : rank_up = py-1, rank_down = py+1
    // => Cart_shift(..., 1, +1, &rank_up, &rank_down) signifie:
    //    rank_up = voisine en py-1
    //    rank_down= voisine en py+1
    MPI_Cart_shift(sd->cart_comm, 0, 1, &rank_left, &rank_right);
    MPI_Cart_shift(sd->cart_comm, 1, 1, &rank_up, &rank_down);

    // ********************
    // 1) Echange "top <-> bottom" en criss-cross
    //    On envoie notre top vers rank_up, on reçoit le bottom de rank_down
    //    => on utilise un tag=101
    // ********************
    MPI_Sendrecv(
        &src[IDX(sd, 1, 1)], 1, sd->row_type,  // top row
        rank_up, 101,
        &src[IDX(sd, 1, sd->local_ny+1)], 1, sd->row_type,  // bottom halo
        rank_down, 101,
        sd->cart_comm, MPI_STATUS_IGNORE
    );

    // ********************
    // 2) Echange "bottom <-> top"
    //    On envoie notre bottom vers rank_down, on reçoit le top de rank_up
    //    => on utilise un tag=102
    // ********************
    MPI_Sendrecv(
        &src[IDX(sd, 1, sd->local_ny)], 1, sd->row_type, // bottom row
        rank_down, 102,
        &src[IDX(sd, 1, 0)], 1, sd->row_type, // top halo
        rank_up, 102,
        sd->cart_comm, MPI_STATUS_IGNORE
    );

    // ********************
    // 3) Echange des colonnes (gauche / droite) -- même principe
    //    a) Envoyer "colonne de gauche" -> rank_left, recevoir "colonne de droite" <- rank_right
    //    b) Envoyer "colonne de droite" -> rank_right, recevoir "colonne de gauche" <- rank_left
    // ********************
    // a) left<->right
    MPI_Sendrecv(
        &src[IDX(sd, 1, 1)], 1, sd->column_type,
        rank_left, 201,
        &src[IDX(sd, sd->local_nx+1, 1)], 1, sd->column_type,
        rank_right, 201,
        sd->cart_comm, MPI_STATUS_IGNORE
    );

    // b) right<->left
    MPI_Sendrecv(
        &src[IDX(sd, sd->local_nx, 1)], 1, sd->column_type,
        rank_right, 202,
        &src[IDX(sd, 0, 1)], 1, sd->column_type,
        rank_left, 202,
        sd->cart_comm, MPI_STATUS_IGNORE
    );
}


/* =========================================================== */
/* 7) Calcul d’un pas : swap, échange halos, mise à jour       */
/*    Retourne 1 si localement convergent, 0 sinon             */
/* =========================================================== */
static int do_step(StencilData *sd)
{
    /* On swap les buffers : on calcule 'values' à partir de 'prev_values' */
    stencil_t* tmp = sd->prev_values;
    sd->prev_values = sd->values;
    sd->values = tmp;

    /* Echange des halos avant le calcul */
    exchange_halos(sd, sd->prev_values);

    /* Calcul au centre */
    int local_conv = 1;
    for (int j = 1; j <= sd->local_ny; j++) {
        for (int i = 1; i <= sd->local_nx; i++) {
            /* Mise à jour (même formule que le code séquentiel) */
            sd->values[IDX(sd, i, j)] =
                alpha * ( sd->prev_values[IDX(sd, i-1, j)]
                        + sd->prev_values[IDX(sd, i+1, j)]
                        + sd->prev_values[IDX(sd, i, j-1)]
                        + sd->prev_values[IDX(sd, i, j+1)] )
                + (1.0f - 4.0f * alpha) * sd->prev_values[IDX(sd, i, j)];

            if ( local_conv &&
                 (fabsf(sd->values[IDX(sd, i, j)] - sd->prev_values[IDX(sd, i, j)]) > epsilon) ) {
                local_conv = 0;
            }
        }
    }

    return local_conv;
}

/* =========================================================== */
/* 8) Affichage global (sur le rang 0)                         */
/*    On envoie (send) les blocs locaux au rang 0 qui reconstruit. */
/* =========================================================== */
static void print_global_array(StencilData *sd)
{
    int px = sd->coords[0];
    int py = sd->coords[1];
    int block_nx = sd->local_nx;
    int block_ny = sd->local_ny;
    int global_nx = sd->global_nx;
    int global_ny = sd->global_ny;

    /* On va envoyer un buffer local (sans halo) de size block_nx*block_ny */
    int local_size = block_nx * block_ny;
    stencil_t* local_buf = (stencil_t*) malloc(local_size * sizeof(stencil_t));
    if (!local_buf) {
        fprintf(stderr, "Erreur d’alloc dans print_global_array\n");
        return;
    }
    /* Copie (sans halo) */
    for (int j = 0; j < block_ny; j++) {
        for (int i = 0; i < block_nx; i++) {
            local_buf[j*block_nx + i] = sd->values[IDX(sd, i+1, j+1)];
        }
    }

    /* Sur le rang 0, on va reconstituer le tableau global. */
    stencil_t* global_buf = NULL;
    if (sd->rank == 0) {
        global_buf = (stencil_t*) malloc(global_nx * global_ny * sizeof(stencil_t));
    }

    /* On envoie d’abord px,py pour savoir où ranger, puis le bloc. */
    if (sd->rank == 0) {
        /* Copie locale d’abord */
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
        printf("\n===== Résultat global final =====\n");
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
/* main : utilise les fonctions ci-dessus                      */
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

    /* Boucle de calcul */
    int s;
    int global_converged = 0;
    for (s = 0; s < stencil_max_steps; s++) {
        int local_conv = do_step(&sd);
        /* Réduction logique AND : si tout le monde converge => global_converged=1 */
        MPI_Allreduce(&local_conv, &global_converged, 1, MPI_INT, MPI_LAND, sd.cart_comm);
        if (global_converged) {
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double elapsed_usec = (t2.tv_sec - t1.tv_sec)*1e6 + (t2.tv_nsec - t1.tv_nsec)/1e3;

    /* Affichage temps & performances sur le rang 0 */
    if (sd.rank == 0) {
        printf("# steps = %d\n", s);
        printf("# time  = %g usecs.\n", elapsed_usec);
        double flop = 6.0 * sd.global_nx * sd.global_ny * s;  /* calcul naïf */
        double gflops = flop / (elapsed_usec * 1000.0);
        printf("# gflops= %g\n", gflops);
    }

    /* (Ré)activez l’affichage global si nécessaire */
    // print_global_array(&sd);

    /* Nettoyage */
    clean_up(&sd);
    return 0;
}

