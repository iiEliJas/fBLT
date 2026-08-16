# ============================================================================
# CONFIGURATION
# ============================================================================
CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Iinclude -Itests -D_POSIX_C_SOURCE=200809L
LDLIBS ?= -lm

# Directories
SRC_DIR := src
TESTS_DIR := tests
RUN_DIR := run
OBJ_DIR := obj
BIN_DIR := bin


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
    $(SRC_DIR)/core/allocator.c \
    $(SRC_DIR)/core/backend.c \
    $(SRC_DIR)/backend_cpu/elementwise_cpu.c \
    $(SRC_DIR)/backend_cpu/linalg_cpu.c \
    $(SRC_DIR)/backend_cpu/reductions_cpu.c \
    $(SRC_DIR)/backend_cpu/optim_cpu.c \
 	$(SRC_DIR)/backend_cpu/mask_builder_cpu.c \
    $(SRC_DIR)/models/entropy.c \
    $(SRC_DIR)/models/byte_embedding.c \
    $(SRC_DIR)/models/patcher.c \
    $(SRC_DIR)/models/attention.c \
    $(SRC_DIR)/models/transformer.c \
    $(SRC_DIR)/models/entropy_lm.c \
	$(SRC_DIR)/models/hash_ngram.c 

TEST_SRCS := \
    $(wildcard $(TESTS_DIR)/unit/*/*.c) \
	$(wildcard $(TESTS_DIR)/parity/*.c) \
    $(TESTS_DIR)/test_helpers.c \
    $(TESTS_DIR)/test_main.c


# ============================================================================
# OBJECT FILES
# ============================================================================
CORE_OBJS := $(addprefix $(OBJ_DIR)/,$(CORE_SRCS:.c=.o))
TEST_OBJS := $(addprefix $(OBJ_DIR)/,$(TEST_SRCS:.c=.o)) $(CORE_OBJS)


# ============================================================================
# TARGETS
# ============================================================================
.PHONY: all test main bench threshold clean info help

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

threshold: $(BIN_DIR)/calibrate_threshold$(EXE_EXT)
	@echo [MAIN] Built successfully: $(BIN_DIR)/calibrate_threshold$(EXE_EXT)


# ============================================================================
# BUILD RULES
# ============================================================================

# Compile src to obj files
$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "[CC] $< -> $@"
	$(MKDIR_P)
	@$(CC) $(CFLAGS) -c $< -o $@

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

# Link calibrate_threshold exe
$(BIN_DIR)/calibrate_threshold$(EXE_EXT): $(RUN_DIR)/calibrate_threshold.c $(CORE_OBJS)
	@echo [LD] Linking main executable: $@
	$(MKDIR_BIN)
	@$(CC) $(CFLAGS) $(RUN_DIR)/calibrate_threshold.c $(CORE_OBJS) -o $@ $(LDLIBS)


# ============================================================================
# CLEAN
# ============================================================================
clean:
	@echo [CLEAN] Removing object directory ...
ifeq ($(DETECTED_OS),Windows)
	@if exist $(OBJ_DIR) $(RM_DIR) $(OBJ_DIR)
	@if exist $(BIN_DIR) $(RM_DIR) $(BIN_DIR)
else
	@$(RM_DIR) $(OBJ_DIR) $(BIN_DIR) 2>/dev/null || true
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
	@echo -  make bench      - Build benchmark exe
	@echo -  make threshold  - Build calibrate_threshold exe
	@echo -  make all        - Same as 'make test'
	@echo -  make clean      - Remove all generated files
	@echo -  make info       - Display build config
	@echo -  make help       - Show this message
	@echo Platform detected: $(DETECTED_OS)