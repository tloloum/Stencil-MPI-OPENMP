BUILD_DIR   = build

CC_SEQ      = gcc
CFLAGS_SEQ  = -Wall -g -O4
LDLIBS_SEQ  = -lm -lrt

MPICC       = mpicc
CFLAGS_MPI  = -Wall -g -O4
LDLIBS_MPI  = -lm -lrt

CC_OMP      = mpicc
CFLAGS_OMP  = -Wall -g -O4 -fopenmp
LDLIBS_OMP  = -lm -lrt

CC_HYB      = mpicc
CFLAGS_HYB  = -Wall -g -O4 -fopenmp
LDLIBS_HYB  = -lm -lrt

TARGET_SEQ     = stencil_seq
TARGET_MPI     = stencil_mpi
TARGET_OMP     = stencil_omp
TARGET_HYBRID  = stencil_hybrid


all: $(BUILD_DIR) \
     $(BUILD_DIR)/$(TARGET_SEQ) \
     $(BUILD_DIR)/$(TARGET_MPI) \
     $(BUILD_DIR)/$(TARGET_OMP) \
     $(BUILD_DIR)/$(TARGET_HYBRID)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


$(BUILD_DIR)/stencil_seq.o: stencil.c
	$(CC_SEQ) $(CFLAGS_SEQ) -c $< -o $@

$(BUILD_DIR)/$(TARGET_SEQ): $(BUILD_DIR)/stencil_seq.o
	$(CC_SEQ) $(CFLAGS_SEQ) -o $@ $^ $(LDLIBS_SEQ)


$(BUILD_DIR)/stencil_mpi.o: stencil_mpi.c
	$(MPICC) $(CFLAGS_MPI) -c $< -o $@

$(BUILD_DIR)/$(TARGET_MPI): $(BUILD_DIR)/stencil_mpi.o
	$(MPICC) $(CFLAGS_MPI) -o $@ $^ $(LDLIBS_MPI)


$(BUILD_DIR)/stencil_omp.o: stencil_omp.c
	$(CC_OMP) $(CFLAGS_OMP) -c $< -o $@

$(BUILD_DIR)/$(TARGET_OMP): $(BUILD_DIR)/stencil_omp.o
	$(CC_OMP) $(CFLAGS_OMP) -o $@ $^ $(LDLIBS_OMP)


$(BUILD_DIR)/stencil_hybrid.o: stencil_hybrid.c
	$(CC_HYB) $(CFLAGS_HYB) -c $< -o $@

$(BUILD_DIR)/$(TARGET_HYBRID): $(BUILD_DIR)/stencil_hybrid.o
	$(CC_HYB) $(CFLAGS_HYB) -o $@ $^ $(LDLIBS_HYB)


clean:
	-rm -rf $(BUILD_DIR)

mrproper: clean
	-rm -f *~


archive: stencil.c stencil_mpi.c stencil_omp.c stencil_hybrid.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil/ )

