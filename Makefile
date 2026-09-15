# ============================================================================
# CONFIGURATION
# ============================================================================
CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Icsrc -Itests
# Header dependency tracking (auto-generated .d files)
DEPFLAGS := -MMD -MP
LDLIBS ?= -lm

# CUDA backend: build with `make CUDA=1 <target>` (Linux only).
# Uses the system toolkit at /usr/local/cuda (must be >= driver version;
# the apt nvcc package shadows /usr/local/cuda/bin/nvcc and produces
# binaries that fail against newer drivers, hence the explicit default).
CUDA ?= 0

# Directories
SRC_DIR := csrc
TESTS_DIR := tests
TOOLS_DIR := csrc/core
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
    # All Windows builds use MSYS2 / MinGW shell — POSIX mkdir works.
    MKDIR_P = @mkdir -p $(dir $@)
    MKDIR_BIN = @mkdir -p $(BIN_DIR)
else
    DETECTED_OS := Linux
    RM := rm -f
    RM_DIR := rm -rf
    EXE_EXT :=
    MKDIR_P = @mkdir -p $(dir $@)
    MKDIR_BIN = @mkdir -p $(BIN_DIR)
endif

# POSIX source feature test macro not for Windows
ifneq ($(OS),Windows_NT)
    CFLAGS += -D_POSIX_C_SOURCE=200809L
endif


# ============================================================================
# SOURCE FILES
# ============================================================================

CORE_SRCS := \
    $(SRC_DIR)/core/tensor.c \
    $(SRC_DIR)/core/json.c \
    $(SRC_DIR)/core/allocator.c \
    $(SRC_DIR)/core/backend.c \
    $(SRC_DIR)/ops/elementwise_cpu.c \
    $(SRC_DIR)/ops/linalg_cpu.c \
    $(SRC_DIR)/ops/reductions_cpu.c \
    $(SRC_DIR)/ops/optim_cpu.c \
 	$(SRC_DIR)/ops/mask_builder_cpu.c \
 	$(SRC_DIR)/ops/patch_pool_cpu.c \
	$(SRC_DIR)/ops/vecmath_cpu.c \
	$(SRC_DIR)/ops/attn_core_cpu.c \
	$(SRC_DIR)/ops/gather_scatter_cpu.c \
	$(SRC_DIR)/ops/row_stats_cpu.c \
	$(SRC_DIR)/ops/cast_cpu.c \
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
	$(SRC_DIR)/models/model_builder.c \
	$(SRC_DIR)/models/checkpoint.c \
	$(SRC_DIR)/infer/stats.c \
	$(SRC_DIR)/infer/rope_gather.c \
	$(SRC_DIR)/infer/kv_cache.c \
	$(SRC_DIR)/infer/block_generation.c \
	$(SRC_DIR)/infer/self_speculation.c \
	$(SRC_DIR)/core/generate_greedy.c \
	$(SRC_DIR)/core/flops.c

# CUDA backend sources (compiled by nvcc when CUDA=1). Kernel files are .cu;
# public signatures live alongside implementations under csrc/ops/.
CUDA_SRCS := \
	$(SRC_DIR)/ops/memory.cu \
	$(SRC_DIR)/ops/vecmath.cu \
	$(SRC_DIR)/ops/elementwise.cu \
	$(SRC_DIR)/ops/reductions.cu \
	$(SRC_DIR)/ops/linalg.cu \
	$(SRC_DIR)/ops/mask_builder.cu \
	$(SRC_DIR)/ops/patch_pool.cu \
	$(SRC_DIR)/ops/optim.cu \
	$(SRC_DIR)/ops/attn_core.cu \
	$(SRC_DIR)/ops/gather_scatter.cu \
	$(SRC_DIR)/ops/row_stats.cu \
	$(SRC_DIR)/ops/cast_cuda.cu
CUDA_SMOKE_SRCS := \
	$(SRC_DIR)/ops/smoke.cu

ifeq ($(CUDA),1)
# GPU architecture — override with: make CUDA=1 NVCC_ARCH="sm_86 sm_89"
NVCC_ARCH ?= sm_89

ifeq ($(OS),Windows_NT)
    # ── Windows CUDA (MSVC host compiler required by nvcc) ──────────
    CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6
    NVCC       = "$(CUDA_PATH)/bin/nvcc.exe"
    NVCC_FLAGS  = -O2 -std=c++17 \
                  $(foreach arch,$(NVCC_ARCH),-gencode arch=compute_$(arch:sm_%=%),code=$(arch)) \
                  -Icsrc -DBLT_WITH_CUDA
    # Dynamic cudart: cudart_static requires MSVC CRT symbols
    # (__GSHandlerCheck etc.) that MinGW's linker cannot resolve.
    CUDA_LIBS  = -L"$(CUDA_PATH)/lib/x64" \
                 -lcublas -lcudart
else
    # ── Linux CUDA ──────────────────────────────────────────────────
    CUDA_PATH ?= /usr/local/cuda
    NVCC       = $(CUDA_PATH)/bin/nvcc
    NVCC_FLAGS  = -O2 -std=c++17 \
                  $(foreach arch,$(NVCC_ARCH),-gencode arch=compute_$(arch:sm_%=%),code=$(arch)) \
                  -Icsrc -DBLT_WITH_CUDA
    # Static cudart avoids libnvidia-ptxjitcompiler version mismatch on WSL2.
    CUDA_LIBS  = -L$(CUDA_PATH)/lib64 \
                 -lcublas -lcudart_static -ldl -lpthread -lrt -lstdc++
endif

CFLAGS += -DBLT_WITH_CUDA
LDLIBS += $(CUDA_LIBS)
# Separate trees so CPU and CUDA objects never mix.
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
ifeq ($(CUDA),1)
CUDA_OBJS := $(addprefix $(OBJ_DIR)/,$(CUDA_SRCS:.cu=.o))
CUDA_SMOKE_OBJS := $(addprefix $(OBJ_DIR)/,$(CUDA_SMOKE_SRCS:.cu=.o))
CORE_OBJS += $(CUDA_OBJS)
endif
BENCH_LIB_SRCS := \
	$(BENCH_DIR)/harness.c
BENCH_LIB_OBJS := $(addprefix $(OBJ_DIR)/,$(BENCH_LIB_SRCS:.c=.o))
TEST_OBJS := $(addprefix $(OBJ_DIR)/,$(TEST_SRCS:.c=.o)) $(CORE_OBJS)


# ============================================================================
# TARGETS
# ============================================================================
.PHONY: all test main bench bench-harness bench-cuda sandbox e2e-dv bench-infer sweep cuda-smoke cuda-sanitize cuda-compile parity-data infer clean info help

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

cuda-compile: $(CUDA_OBJS)
	@echo "All CUDA sources compiled successfully."

parity-data:
ifeq ($(wildcard data/tests),)
	@echo [PARITY] Generating golden test data...
	python3 tests/py/parity_generate.py
else
	@echo [PARITY] Golden test data already exists, skipping...
endif

test: $(BIN_DIR)/test_main$(EXE_EXT) parity-data
	@echo [TEST] Running tests...
	@./$(BIN_DIR)/test_main$(EXE_EXT)

test-asan:
	@echo [TEST-ASAN] Building with sanitizers...
	@$(MAKE) clean
	@$(MAKE) CC=gcc CFLAGS="-O1 -g -std=c99 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -Icsrc -Itests -D_POSIX_C_SOURCE=200809L" LDLIBS="-lm" $(BIN_DIR)/test_main
	@echo [TEST-ASAN] Running tests (leak detection off)...
	@ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 ./$(BIN_DIR)/test_main$(EXE_EXT)

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

infer: $(BIN_DIR)/infer$(EXE_EXT)
	@echo [INFER] Built successfully: $(BIN_DIR)/infer$(EXE_EXT)

train-blt-d: $(BIN_DIR)/train_blt_d$(EXE_EXT)
	@echo [TRAIN] Built successfully: $(BIN_DIR)/train_blt_d$(EXE_EXT)
	
# sanitizer gate: run the full CUDA test suite (includes the
# training-step parity) under compute-sanitizer. Requires a native Linux
# box: WSL2/dxg devices are rejected by the sanitizer ("Device not
# supported"). TOOL selects memcheck (default), racecheck, initcheck, or
# synccheck.
# Windows: compute-sanitizer is not available; gate behind non-Windows.
ifneq ($(OS),Windows_NT)
CUDA_SANITIZER ?= /usr/local/cuda/bin/compute-sanitizer
SANITIZE_TOOL  ?= memcheck
.PHONY: cuda-sanitize
cuda-sanitize: $(BIN_DIR)/test_main$(EXE_EXT)
	@echo [SANITIZE] $(SANITIZE_TOOL) over the full CUDA suite...
	@$(CUDA_SANITIZER) --target-processes all --tool $(SANITIZE_TOOL) ./$(BIN_DIR)/test_main$(EXE_EXT)
endif


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
$(OBJ_DIR)/$(SRC_DIR)/ops/%.o: $(SRC_DIR)/ops/%.cu
	@echo "[NVCC] $< -> $@"
	$(MKDIR_P)
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif

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
$(BIN_DIR)/main$(EXE_EXT): $(SRC_DIR)/main.c $(CORE_OBJS)
	@echo [LD] Linking main executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(SRC_DIR)/main.c $(CORE_OBJS) -o $@ $(LDLIBS)

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
$(BIN_DIR)/sandbox$(EXE_EXT): $(SRC_DIR)/sandbox.c $(CORE_OBJS) $(TOOLS_OBJS)
	@echo [LD] Linking main executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(SRC_DIR)/sandbox.c $(CORE_OBJS) -o $@ $(LDLIBS)

# Link sweep trainer exe
$(BIN_DIR)/train_sweep$(EXE_EXT): $(SRC_DIR)/core/train_sweep.c $(CORE_OBJS) $(BENCH_LIB_OBJS)
	@echo [SWEEP] Built successfully: $(BIN_DIR)/train_sweep$(EXE_EXT)
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) -I$(BENCH_DIR) $(SRC_DIR)/core/train_sweep.c $(CORE_OBJS) $(BENCH_LIB_OBJS) -o $@ $(LDLIBS)

# Link BLT-D trainer exe
$(BIN_DIR)/train_blt_d$(EXE_EXT): $(SRC_DIR)/train_blt_d.c $(CORE_OBJS)
	@echo [LD] Linking BLT-D trainer executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(SRC_DIR)/train_blt_d.c $(CORE_OBJS) -o $@ $(LDLIBS)

# Link CUDA smoke test (CUDA=1 only)
$(BIN_DIR)/cuda_smoke$(EXE_EXT): $(SRC_DIR)/cuda_smoke.c $(CORE_OBJS) $(CUDA_SMOKE_OBJS)
	@echo [LD] Linking CUDA smoke executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(SRC_DIR)/cuda_smoke.c $(CORE_OBJS) $(CUDA_SMOKE_OBJS) -o $@ $(LDLIBS)

# Link inference binary
$(BIN_DIR)/infer$(EXE_EXT): $(SRC_DIR)/infer.c $(CORE_OBJS)
	$(MKDIR_BIN)
	@echo "[CC] $< -> $@"
	@$(CC) $(CFLAGS) $(SRC_DIR)/infer.c $(CORE_OBJS) -o $@ $(LDLIBS)

# Link end-to-end driver
$(BIN_DIR)/e2e_blt_dv$(EXE_EXT): $(SRC_DIR)/e2e_blt_dv.c $(CORE_OBJS)
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(SRC_DIR)/e2e_blt_dv.c $(CORE_OBJS) -o $@ $(LDLIBS)

# Link inference benchmark
$(BIN_DIR)/infer_bench$(EXE_EXT): bench/infer_bench.c $(CORE_OBJS) $(BENCH_LIB_OBJS)
	$(MKDIR_BIN)
	@echo "[CC] $< -> $@"
	@$(CC) $(CFLAGS) -I$(BENCH_DIR) bench/infer_bench.c $(CORE_OBJS) $(BENCH_LIB_OBJS) -o $@ $(LDLIBS)

$(BIN_DIR)/cuda_bench$(EXE_EXT): bench/cuda_bench.c $(CORE_OBJS) $(BENCH_LIB_OBJS)
	$(MKDIR_BIN)
	@echo "[CC] $< -> $@"
	@$(CC) $(CFLAGS) -I$(BENCH_DIR) bench/cuda_bench.c $(CORE_OBJS) $(BENCH_LIB_OBJS) -o $@ $(LDLIBS)

	
# ============================================================================
# CLEAN
# ============================================================================
clean:
	@echo [CLEAN] Removing object and binary directories ...
	@$(RM_DIR) obj obj-cuda bin bin-cuda 2>/dev/null || true
	@echo [CLEAN] Complete.


# ============================================================================
# HELP
# ============================================================================
help:
	@echo --- BLT Project Makefile ---
	@echo - Available targets:
	@echo -  make test       - Build and run test exe
	@echo -  make parity-data - Generate golden test data (if missing)
	@echo -  make main       - Build main exe
	@echo -  make bench       - Build benchmark exe
	@echo -  make bench-harness - Build + run benchmark harness self-test
	@echo -  make infer      - Build inference binary
	@echo -  make train-blt-d - Build BLT-D trainer
	@echo -  make all        - Same as 'make test'
	@echo -  make clean      - Remove all generated files
	@echo -  make info       - Display build config
	@echo -  make help       - Show this message
	@echo -  make CUDA=1 ... - Enable the CUDA backend (Linux + nvcc required)
	@echo -  make CUDA=1 cuda-smoke - Device sanity check
	@echo -  make CUDA=1 cuda-sanitize [SANITIZE_TOOL=memcheck|racecheck|initcheck|synccheck]
	@echo -       - Full CUDA test suite under compute-sanitizer (native box only)
	@echo Platform detected: $(DETECTED_OS)
-include $(wildcard obj/*.d obj-cuda/*.d)



