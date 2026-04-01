# =============================================================================
# GPU Computing 2026 — SpMV Deliverable 1
# =============================================================================

CC       = gcc
NVCC     = nvcc
OPT      = -std=c99 -O3
GPU_ARCH = --gpu-architecture=sm_80
GPU_FLAGS = -m64

# Directory layout
TIMER_SRC  := TIMER_LIB/src
TIMER_INC  := TIMER_LIB/include
TIMER_OBJ  := TIMER_LIB/obj
INC_DIR    := include
CPU_DIR    := CPU
GPU_DIR    := GPU
BIN_DIR    := bin
CPU_BIN    := $(BIN_DIR)/CPU
GPU_BIN    := $(BIN_DIR)/GPU
BATCH_OUT  := outputs
DATA_DIR   := Data
RESULTS    := results_tables
SCRIPTS    := scripts

# Timer library object
TIMER_LIB_OBJ := $(TIMER_OBJ)/my_time_lib.o

# Auto-discover sources
CPU_SRCS := $(wildcard $(CPU_DIR)/*.c)
GPU_SRCS := $(wildcard $(GPU_DIR)/*.cu)

# Derive binary names
CPU_BINS := $(patsubst $(CPU_DIR)/%.c,   $(CPU_BIN)/%,      $(CPU_SRCS))
GPU_BINS := $(patsubst $(GPU_DIR)/%.cu,  $(GPU_BIN)/%.exec, $(GPU_SRCS))

# =============================================================================
# Default target
# =============================================================================
.PHONY: all cpu gpu data \
        clean_bin clean_outputs clean_results clean \
        help

all: cpu gpu

cpu: $(CPU_BINS)

gpu: $(GPU_BINS)

# =============================================================================
# Data download
# =============================================================================
data:
	@mkdir -p $(BATCH_OUT) $(RESULTS) assets
	bash $(SCRIPTS)/download_data.sh

# =============================================================================
# Timer library
# =============================================================================
$(TIMER_LIB_OBJ): $(TIMER_SRC)/my_time_lib.c
	@mkdir -p $(TIMER_OBJ) $(BIN_DIR) $(CPU_BIN) $(BATCH_OUT)
	$(CC) -c $< -o $@ -I$(TIMER_INC) $(OPT)

# =============================================================================
# CPU binaries
# =============================================================================
$(CPU_BIN)/%: $(CPU_DIR)/%.c $(TIMER_LIB_OBJ)
	@mkdir -p $(CPU_BIN)
	$(CC) $< -o $@ $(TIMER_LIB_OBJ) -I$(TIMER_INC) -I$(INC_DIR) $(OPT) -lm

# =============================================================================
# GPU binaries  (requires: module load CUDA/12.1.1 before running make gpu)
# =============================================================================
$(GPU_BIN)/%.exec: $(GPU_DIR)/%.cu
	@mkdir -p $(GPU_BIN)
	$(NVCC) $(GPU_ARCH) $(GPU_FLAGS) -I$(TIMER_INC) -I$(INC_DIR) -o $@ $<

# =============================================================================
# Clean targets
# =============================================================================
clean_bin:
	rm -rf $(BIN_DIR) $(TIMER_OBJ)

clean_outputs:
	rm -f $(BATCH_OUT)/*.out $(BATCH_OUT)/*.err $(BATCH_OUT)/*.txt

clean_results:
	rm -f $(RESULTS)/*.csv assets/*.png assets/best_gpu_config.sh

clean: clean_bin clean_outputs clean_results
	rm -rf $(DATA_DIR)

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "GPU Computing 2026 — SpMV Deliverable 1"
	@echo ""
	@echo "Build targets:"
	@echo "  all            build CPU + GPU kernels (default)"
	@echo "  cpu            build CPU kernels only"
	@echo "  gpu            build GPU kernels only  [needs: module load CUDA/12.1.1]"
	@echo ""
	@echo "Data:"
	@echo "  data           download all matrices from SuiteSparse into Data/"
	@echo ""
	@echo "Clean targets:"
	@echo "  clean_bin      remove compiled binaries and object files"
	@echo "  clean_outputs  remove SLURM output files (outputs/*.txt, *.err)"
	@echo "  clean_results  remove CSVs (results_tables/) and plots (assets/)"
	@echo "  clean          remove everything: binaries + outputs + results + Data/"
	@echo ""
