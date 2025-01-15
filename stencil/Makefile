##############################################################################
# Variables de compilation
##############################################################################
# On utilise mpicc (qui est un wrapper autour de gcc/clang avec les bonnes libs MPI).
CC      = mpicc

# Options de compilation : -Wall pour les avertissements, -g pour le debug, -O4 pour l’optimisation
CFLAGS += -Wall -g -O4

# Bibliothèques à l’édition de liens : math et real-time
LDLIBS += -lm -lrt

# Nom du binaire à générer
TARGET  = stencil_mpi

##############################################################################
# Règles "par défaut"
##############################################################################
# all dépend de la cible principale $(TARGET).
all: $(TARGET)

# Règle pour produire le binaire à partir des .o
$(TARGET): stencil_mpi.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Règle générique de compilation .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $<

##############################################################################
# Règles de nettoyage
##############################################################################
# Supprime le binaire et les fichiers objets
clean:
	-rm -f $(TARGET) *.o

# Supprime en plus les fichiers temporaires
mrproper: clean
	-rm -f *~

##############################################################################
# Archivage du code (si besoin)
##############################################################################
# Crée une archive .tar.gz contenant le code et le Makefile
archive: stencil_mpi.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil/ )

