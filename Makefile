# ============================================================================
# CONFIGURATION
# ============================================================================
CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Iinclude -Itests -Itools -D_POSIX_C_SOURCE=200809L
# Header dependency tracking (auto-generated .d files)
DEPFLAGS := -MMD -MP
LDLIBS ?= -lm

# CUDA backend: build with `make CUDA=1 <target>` (Linux only).
# Uses the system toolkit at /usr/local/cuda (must be >= driver version;
# the apt nvcc package shadows /usr/local/cuda/bin/nvcc and produces
# binaries that fail against newer drivers, hence the explicit default).
CUDA ?= 0

# Directories
SRC_DIR := src
TESTS_DIR := tests
TOOLS_DIR := tools
RUN_DIR := run
OBJ_DIR := obj
BIN_DIR := bin
BENCH_DIR := bench


# ============================================================================
# PLATFORM DETECTION
# ============================================================================
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM := del /Q
    RM_DIR := rmdir /S /Q
    EXE_EXT := .exe
    # Windows command to create parent directory
    MKDIR_P = @if not exist "$(subst /,\,$(dir $@))" cmd /c "mkdir $(subst /,\,$(dir $@))"
    MKDIR_BIN = @if not exist "$(BIN_DIR)" cmd /c "mkdir $(BIN_DIR)"
else
    DETECTED_OS := Linux
    RM := rm -f
    RM_DIR := rm -rf
    EXE_EXT :=
    MKDIR_P = @mkdir -p $(dir $@)
    MKDIR_BIN = @mkdir -p $(BIN_DIR)
endif


# ============================================================================
# SOURCE FILES
# ============================================================================

CORE_SRCS := \
    $(SRC_DIR)/core/tensor.c \
    $(SRC_DIR)/core/json.c \
    $(SRC_DIR)/core/allocator.c \
    $(SRC_DIR)/core/backend.c \
    $(SRC_DIR)/backend_cpu/elementwise_cpu.c \
    $(SRC_DIR)/backend_cpu/linalg_cpu.c \
    $(SRC_DIR)/backend_cpu/reductions_cpu.c \
    $(SRC_DIR)/backend_cpu/optim_cpu.c \
 	$(SRC_DIR)/backend_cpu/mask_builder_cpu.c \
 	$(SRC_DIR)/backend_cpu/patch_pool_cpu.c \
	$(SRC_DIR)/backend_cpu/vecmath_cpu.c \
	$(SRC_DIR)/backend_cpu/attn_core_cpu.c \
	$(SRC_DIR)/backend_cpu/gather_scatter_cpu.c \
	$(SRC_DIR)/backend_cpu/row_stats_cpu.c \
    $(SRC_DIR)/models/entropy.c \
    $(SRC_DIR)/models/byte_embedding.c \
    $(SRC_DIR)/models/patcher.c \
    $(SRC_DIR)/models/attention.c \
	$(SRC_DIR)/models/cross_attention.c \
    $(SRC_DIR)/models/transformer.c \
	$(SRC_DIR)/models/transformer_stack.c \
    $(SRC_DIR)/models/entropy_lm.c \
	$(SRC_DIR)/models/hash_ngram.c \
 	$(SRC_DIR)/models/local_common.c \
 	$(SRC_DIR)/models/local_encoder.c \
	$(SRC_DIR)/models/local_decoder.c \
	$(SRC_DIR)/models/block_diffusion.c \
	$(SRC_DIR)/models/global_transformer.c \
	$(SRC_DIR)/models/model.c \
	$(SRC_DIR)/models/checkpoint.c \
	$(SRC_DIR)/infer/stats.c \
	$(SRC_DIR)/infer/rope_gather.c \
	$(SRC_DIR)/infer/kv_cache.c \
	$(SRC_DIR)/infer/block_generation.c \
	$(SRC_DIR)/infer/self_speculation.c
TOOLS_SRCS := \
	$(TOOLS_DIR)/generate_greedy.c \
	$(TOOLS_DIR)/flops.c \

# CUDA backend sources (compiled by nvcc when CUDA=1). Kernel files are .cu;
# public signatures stay in the shared headers under include/blt/.
CUDA_SRCS := \
	$(SRC_DIR)/backend_cuda/memory.cu \
	$(SRC_DIR)/backend_cuda/vecmath.cu \
	$(SRC_DIR)/backend_cuda/elementwise.cu \
	$(SRC_DIR)/backend_cuda/reductions.cu \
	$(SRC_DIR)/backend_cuda/linalg.cu \
	$(SRC_DIR)/backend_cuda/mask_builder.cu \
	$(SRC_DIR)/backend_cuda/patch_pool.cu \
	$(SRC_DIR)/backend_cuda/optim.cu \
	$(SRC_DIR)/backend_cuda/attn_core.cu \
	$(SRC_DIR)/backend_cuda/gather_scatter.cu \
	$(SRC_DIR)/backend_cuda/row_stats.cu
CUDA_SMOKE_SRCS := \
	$(SRC_DIR)/backend_cuda/smoke.cu

ifeq ($(CUDA),1)
NVCC ?= /usr/local/cuda/bin/nvcc
NVCC_FLAGS := -O2 -std=c++17 -gencode arch=compute_89,code=sm_89 \
              -Iinclude -DBLT_WITH_CUDA -MMD -MP
# Static cudart is required here: the shared libcudart.so.13 init path picks
# up the system-installed (older) libnvidia-ptxjitcompiler under WSL2 and
# segfaults; the static runtime avoids that dependency entirely.
CUDA_LIBS := -L/usr/local/cuda/lib64 -lcublas -lcudart_static -ldl -lpthread -lrt -lstdc++
CFLAGS += -DBLT_WITH_CUDA
LDLIBS += $(CUDA_LIBS)
# Separate trees so CPU and CUDA objects never mix: core sources are
# compiled with -DBLT_WITH_CUDA only in the CUDA tree.
OBJ_DIR := obj-cuda
BIN_DIR := bin-cuda
endif

TEST_SRCS := \
    $(wildcard $(TESTS_DIR)/unit/*/*.c) \
	$(wildcard $(TESTS_DIR)/parity/*.c) \
    $(TESTS_DIR)/test_helpers.c \
    $(TESTS_DIR)/test_main.c


# ============================================================================
# OBJECT FILES
# ============================================================================
CORE_OBJS := $(addprefix $(OBJ_DIR)/,$(CORE_SRCS:.c=.o))
TOOLS_OBJS := $(addprefix $(OBJ_DIR)/,$(TOOLS_SRCS:.c=.o))
ifeq ($(CUDA),1)
CUDA_OBJS := $(addprefix $(OBJ_DIR)/,$(CUDA_SRCS:.cu=.o))
CUDA_SMOKE_OBJS := $(addprefix $(OBJ_DIR)/,$(CUDA_SMOKE_SRCS:.cu=.o))
CORE_OBJS += $(CUDA_OBJS)
endif
BENCH_LIB_SRCS := \
	$(BENCH_DIR)/harness.c
BENCH_LIB_OBJS := $(addprefix $(OBJ_DIR)/,$(BENCH_LIB_SRCS:.c=.o))
TEST_OBJS := $(addprefix $(OBJ_DIR)/,$(TEST_SRCS:.c=.o)) $(CORE_OBJS) $(TOOLS_OBJS)


# ============================================================================
# TARGETS
# ============================================================================
.PHONY: all test main bench bench-harness bench-cuda sandbox e2e-dv bench-infer sweep cuda-smoke cuda-sanitize clean info help

all: test

info:
	@echo ================================
	@echo Detected OS: $(DETECTED_OS)
	@echo ================================
	@echo CC: $(CC)
	@echo CFLAGS: $(CFLAGS)
	@echo LDLIBS: $(LDLIBS)
	@echo OBJ_DIR: $(OBJ_DIR)
	@echo BIN_DIR: $(BIN_DIR)
	@echo ================================

test: $(BIN_DIR)/test_main$(EXE_EXT)
	@echo [TEST] Running tests...
	@./$(BIN_DIR)/test_main$(EXE_EXT)

main: $(BIN_DIR)/main$(EXE_EXT)
	@echo [MAIN] Built successfully: $(BIN_DIR)/main$(EXE_EXT)

bench: $(BIN_DIR)/bench_patcher$(EXE_EXT)
	@echo [BENCH] Built successfully: $(BIN_DIR)/bench$(EXE_EXT)

bench-infer: $(BIN_DIR)/infer_bench$(EXE_EXT)
	@echo [BENCH] Built successfully: $(BIN_DIR)/infer_bench$(EXE_EXT)
	@./$(BIN_DIR)/infer_bench$(EXE_EXT)

bench-cuda: $(BIN_DIR)/cuda_bench$(EXE_EXT)
	@echo [BENCH] Running CUDA/CPU perf benchmark...
	@./$(BIN_DIR)/cuda_bench$(EXE_EXT)

bench-harness: $(BIN_DIR)/bench_harness_selftest$(EXE_EXT)
	@echo [BENCH] Running harness self-test...
	@./$(BIN_DIR)/bench_harness_selftest$(EXE_EXT) bench/dummy_results.jsonl

sandbox: $(BIN_DIR)/sandbox$(EXE_EXT)
	@echo [MAIN] Built successfully: $(BIN_DIR)/sandbox$(EXE_EXT)
	@./$(BIN_DIR)/sandbox$(EXE_EXT)

e2e-dv: $(BIN_DIR)/e2e_blt_dv$(EXE_EXT)
	@echo [MAIN] Built successfully: $(BIN_DIR)/e2e_blt_dv$(EXE_EXT)
	@./$(BIN_DIR)/e2e_blt_dv$(EXE_EXT)

sweep: $(BIN_DIR)/train_sweep$(EXE_EXT)
	@echo [SWEEP] Built successfully: $(BIN_DIR)/train_sweep$(EXE_EXT)

cuda-smoke: $(BIN_DIR)/cuda_smoke$(EXE_EXT)
	@echo [CUDA] Built successfully: $(BIN_DIR)/cuda_smoke$(EXE_EXT)
	@./$(BIN_DIR)/cuda_smoke$(EXE_EXT)

# Stage-4 sanitizer gate: run the full CUDA test suite (includes the
# training-step parity) under compute-sanitizer. Requires a native Linux
# box: WSL2/dxg devices are rejected by the sanitizer ("Device not
# supported"). TOOL selects memcheck (default), racecheck, initcheck, or
# synccheck.
CUDA_SANITIZER ?= /usr/local/cuda/bin/compute-sanitizer
SANITIZE_TOOL  ?= memcheck
.PHONY: cuda-sanitize
cuda-sanitize: $(BIN_DIR)/test_main$(EXE_EXT)
	@echo [SANITIZE] $(SANITIZE_TOOL) over the full CUDA suite...
	@$(CUDA_SANITIZER) --target-processes all --tool $(SANITIZE_TOOL) ./$(BIN_DIR)/test_main$(EXE_EXT)


# ============================================================================
# BUILD RULES
# ============================================================================

# Compile src to obj files
$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "[CC] $< -> $@"
	$(MKDIR_P)
	@$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# Compile CUDA backend sources to obj files (CUDA=1 only)
ifeq ($(CUDA),1)
$(OBJ_DIR)/$(SRC_DIR)/backend_cuda/%.o: $(SRC_DIR)/backend_cuda/%.cu
	@echo "[NVCC] $< -> $@"
	$(MKDIR_P)
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

# Compile tool to obj files
$(OBJ_DIR)/$(TOOLS_DIR)/%.o: $(TOOLS_DIR)/%.c
	@echo "[CC] $< -> $@"
	$(MKDIR_P)
	@$(CC) $(CFLAGS) -c $< -o $@

# Compile bench harness to obj files
$(OBJ_DIR)/$(BENCH_DIR)/%.o: $(BENCH_DIR)/%.c
	@echo "[CC] $< -> $@"
	$(MKDIR_P)
	@$(CC) $(CFLAGS) -I$(BENCH_DIR) -c $< -o $@

# Compile test to obj files
$(OBJ_DIR)/$(TESTS_DIR)/%.o: $(TESTS_DIR)/%.c
	@echo "[CC] $< -> $@"
	$(MKDIR_P)
	@$(CC) $(CFLAGS) -c $< -o $@

# Link test exe
$(BIN_DIR)/test_main$(EXE_EXT): $(TEST_OBJS)
	@echo [LD] Linking test executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(TEST_OBJS) -o $@ $(LDLIBS)

# Link main exe
$(BIN_DIR)/main$(EXE_EXT): $(RUN_DIR)/main.c $(CORE_OBJS)
	@echo [LD] Linking main executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(RUN_DIR)/main.c $(CORE_OBJS) -o $@ $(LDLIBS)

# Link bench exe
$(BIN_DIR)/bench_patcher$(EXE_EXT): $(TESTS_DIR)/bench/bench_patcher.c $(CORE_OBJS)
	@echo [LD] Linking bench executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(TESTS_DIR)/bench/bench_patcher.c $(CORE_OBJS) -o $@ $(LDLIBS)

# Link harness self-test exe
$(BIN_DIR)/bench_harness_selftest$(EXE_EXT): $(TESTS_DIR)/bench/bench_harness_selftest.c $(BENCH_LIB_OBJS)
	@echo [LD] Linking harness self-test executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) -I. $(TESTS_DIR)/bench/bench_harness_selftest.c $(BENCH_LIB_OBJS) -o $@ $(LDLIBS)

# Link sandbox exe
$(BIN_DIR)/sandbox$(EXE_EXT): $(RUN_DIR)/sandbox.c $(CORE_OBJS) $(TOOLS_OBJS)
	@echo [LD] Linking main executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(RUN_DIR)/sandbox.c $(CORE_OBJS) $(TOOLS_OBJS) -o $@ $(LDLIBS)

# Link sweep trainer exe
$(BIN_DIR)/train_sweep$(EXE_EXT): $(TOOLS_DIR)/train_sweep.c $(CORE_OBJS) $(TOOLS_OBJS) $(BENCH_LIB_OBJS)
	@echo [LD] Linking trainer executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) -I$(BENCH_DIR) $(TOOLS_DIR)/train_sweep.c $(CORE_OBJS) $(TOOLS_OBJS) $(BENCH_LIB_OBJS) -o $@ $(LDLIBS)

# Link BLT-D trainer exe (Phase D)
$(BIN_DIR)/train_blt_d$(EXE_EXT): $(RUN_DIR)/train_blt_d.c $(CORE_OBJS)
	@echo [LD] Linking BLT-D trainer executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(RUN_DIR)/train_blt_d.c $(CORE_OBJS) -o $@ $(LDLIBS)

# Link CUDA smoke test (CUDA=1 only)
$(BIN_DIR)/cuda_smoke$(EXE_EXT): $(RUN_DIR)/cuda_smoke.c $(CORE_OBJS) $(CUDA_SMOKE_OBJS)
	@echo [LD] Linking CUDA smoke executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(RUN_DIR)/cuda_smoke.c $(CORE_OBJS) $(CUDA_SMOKE_OBJS) -o $@ $(LDLIBS)

train-blt-d: $(BIN_DIR)/train_blt_d$(EXE_EXT)
	@echo [TRAIN] Built successfully: $(BIN_DIR)/train_blt_d$(EXE_EXT)


# ============================================================================
# CLEAN
# ============================================================================
clean:
	@echo [CLEAN] Removing object and binary directories ...
ifeq ($(DETECTED_OS),Windows)
	@if exist $(OBJ_DIR) $(RM_DIR) $(OBJ_DIR)
	@if exist $(BIN_DIR) $(RM_DIR) $(BIN_DIR)
else
	@$(RM_DIR) obj obj-cuda bin bin-cuda 2>/dev/null || true
endif
	@echo [CLEAN] Complete.


# ============================================================================
# HELP
# ============================================================================
help:
	@echo --- BLT Project Makefile ---
	@echo - Available targets:
	@echo -  make test       - Build and run test exe
	@echo -  make main       - Build main exe
	@echo -  make bench       - Build benchmark exe
	@echo -  make bench-harness - Build + run benchmark harness self-test
	@echo -  make all        - Same as 'make test'
	@echo -  make clean      - Remove all generated files
	@echo -  make info       - Display build config
	@echo -  make help       - Show this message
	@echo -  make CUDA=1 ... - Enable the CUDA backend (Linux + nvcc required)
	@echo -  make CUDA=1 cuda-smoke - Device sanity check
	@echo -  make CUDA=1 cuda-sanitize [SANITIZE_TOOL=memcheck|racecheck|initcheck|synccheck]
	@echo -       - Full CUDA test suite under compute-sanitizer (native box only)
	@echo Platform detected: $(DETECTED_OS)
-include $(shell find obj obj-cuda -name '*.d' 2>/dev/null)


# Link Phase E end-to-end driver
$(BIN_DIR)/e2e_blt_dv$(EXE_EXT): $(RUN_DIR)/e2e_blt_dv.c $(CORE_OBJS)
	@mkdir -p $(BIN_DIR)
	@$(CC) $(CFLAGS) $(RUN_DIR)/e2e_blt_dv.c $(CORE_OBJS) $(TOOLS_OBJS) -o $@ $(LDLIBS)

# Link Phase F inference benchmark
$(BIN_DIR)/infer_bench$(EXE_EXT): bench/infer_bench.c $(CORE_OBJS) $(TOOLS_OBJS) $(BENCH_LIB_OBJS)
	@mkdir -p $(BIN_DIR)
	@echo "[CC] $< -> $@"
	@$(CC) $(CFLAGS) -I$(BENCH_DIR) bench/infer_bench.c $(CORE_OBJS) $(TOOLS_OBJS) $(BENCH_LIB_OBJS) -o $@ $(LDLIBS)

$(BIN_DIR)/cuda_bench$(EXE_EXT): bench/cuda_bench.c $(CORE_OBJS) $(TOOLS_OBJS) $(BENCH_LIB_OBJS)
	@mkdir -p $(BIN_DIR)
	@echo "[CC] $< -> $@"
	@$(CC) $(CFLAGS) -I$(BENCH_DIR) bench/cuda_bench.c $(CORE_OBJS) $(TOOLS_OBJS) $(BENCH_LIB_OBJS) -o $@ $(LDLIBS)
