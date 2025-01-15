##############################################################################
# Variables de compilation
##############################################################################

# mpicc = wrapper autour de gcc ou clang pour inclure MPI
MPICC = mpicc

# Options de compilation communes (avertissements, debug, optimisation)
CFLAGS_COMMON = -Wall -g -O4

# On ajoute -fopenmp pour les versions OpenMP ou hybrides
CFLAGS_OMP = $(CFLAGS_COMMON) -fopenmp

# Bibliothèques à l’édition de liens : math et temps réel
LDLIBS = -lm -lrt

# Noms des exécutables
TARGET_MPI    = stencil_mpi
TARGET_OMP    = stencil_omp
TARGET_HYBRID = stencil_hybrid

##############################################################################
# Règles "par défaut"
##############################################################################
# Par défaut, on compile tout
all: $(TARGET_MPI) $(TARGET_OMP) $(TARGET_HYBRID)

##############################################################################
# 1) Cible MPI "pure"
##############################################################################
# Supposons que le code source s'appelle stencil_mpi.c
stencil_mpi.o: stencil_mpi.c
	$(MPICC) $(CFLAGS_COMMON) -c $<

$(TARGET_MPI): stencil_mpi.o
	$(MPICC) $(CFLAGS_COMMON) -o $@ $^ $(LDLIBS)

##############################################################################
# 2) Cible OpenMP "pure"
##############################################################################
# Supposons que le code source s'appelle stencil_omp.c
stencil_omp.o: stencil_omp.c
	$(MPICC) $(CFLAGS_OMP) -c $<

$(TARGET_OMP): stencil_omp.o
	$(MPICC) $(CFLAGS_OMP) -o $@ $^ $(LDLIBS)

##############################################################################
# 3) Cible Hybride MPI + OpenMP
##############################################################################
# Supposons que le code source s'appelle stencil_hybrid.c
stencil_hybrid.o: stencil_hybrid.c
	$(MPICC) $(CFLAGS_OMP) -c $<

$(TARGET_HYBRID): stencil_hybrid.o
	$(MPICC) $(CFLAGS_OMP) -o $@ $^ $(LDLIBS)

##############################################################################
# Règles de nettoyage
##############################################################################
clean:
	-rm -f *.o $(TARGET_MPI) $(TARGET_OMP) $(TARGET_HYBRID)

mrproper: clean
	-rm -f *~

##############################################################################
# Archivage du code (optionnel)
##############################################################################
# Ex: on suppose que tout est dans le répertoire stencil/
archive: stencil_mpi.c stencil_omp.c stencil_hybrid.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil/ )

