# ============================================================================
# CONFIGURATION
# ============================================================================
CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Iinclude -Itests
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
    MKDIR := mkdir
    SEP := \\
    EXE_EXT := .exe
else
    DETECTED_OS := Linux
    RM := rm -f
    RM_DIR := rm -rf
    MKDIR := mkdir -p
    SEP := /
    EXE_EXT :=
endif


# ============================================================================
# SOURCE FILES
# ============================================================================
# Core src
CORE_SRCS := \
    $(SRC_DIR)/core/tensor.c \
    $(SRC_DIR)/core/allocator.c \
    $(SRC_DIR)/core/backend.c \
    $(SRC_DIR)/backend_cpu/elementwise_cpu.c \
    $(SRC_DIR)/backend_cpu/linalg_cpu.c \
    $(SRC_DIR)/backend_cpu/reductions_cpu.c \
    $(SRC_DIR)/backend_cpu/optim_cpu.c \
    $(SRC_DIR)/models/entropy.c \
    $(SRC_DIR)/models/byte_embedding.c \
    $(SRC_DIR)/models/patcher.c \
    $(SRC_DIR)/models/attention.c \
    $(SRC_DIR)/models/transformer.c \
    $(SRC_DIR)/models/entropy_lm.c

# Test src
TEST_SRCS := \
    $(wildcard $(TESTS_DIR)/unit/*/*.c) \
    $(TESTS_DIR)/test_helpers.c \
    $(TESTS_DIR)/test_main.c \
    $(TESTS_DIR)/integration/test_backend.c \
    $(TESTS_DIR)/integration/test_transformer.c \
    $(TESTS_DIR)/integration/test_entropy_lm.c


# ============================================================================
# OBJECT FILES
# ============================================================================
# Object files for core src
CORE_OBJS := $(addprefix $(OBJ_DIR)/,$(CORE_SRCS:.c=.o))

# Object files for tests
TEST_OBJS := $(addprefix $(OBJ_DIR)/,$(TEST_SRCS:.c=.o)) $(CORE_OBJS)


# ============================================================================
# TARGETS
# ============================================================================
.PHONY: all test main clean info

# Default target
all: test

# Info target
info:
	@echo "================================"
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "================================"
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDLIBS: $(LDLIBS)"
	@echo "OBJ_DIR: $(OBJ_DIR)"
	@echo "BIN_DIR: $(BIN_DIR)"
	@echo "================================"

# Test exe
test: $(BIN_DIR)/test_main$(EXE_EXT)
	@echo [TEST] Running tests...
	@./$(BIN_DIR)/test_main$(EXE_EXT)

# Main exe
main: $(BIN_DIR)/main$(EXE_EXT)
	@echo [MAIN] Built successfully: $(BIN_DIR)/main$(EXE_EXT)

# ============================================================================
# BUILD RULES
# ============================================================================
# Create dir
$(OBJ_DIR):
	@$(MKDIR) $@

$(BIN_DIR):
	@$(MKDIR) $@


ifeq ($(DETECTED_OS),Windows)
    # Helper to convert forward slashes to backslashes
    MKDIR_P = $(shell if not exist "$(subst /,\,$(patsubst %/,%,$(dir $@)))" mkdir "$(subst /,\,$(patsubst %/,%,$(dir $@)))")
else
    MKDIR_P = mkdir -p $(dir $@)
endif

# Compile src to obj files
$(OBJ_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	@echo [CC] $< -> $@
	@$(MKDIR_P)
	@$(CC) $(CFLAGS) -c $< -o $@

# Compile test to obj files
$(OBJ_DIR)/$(TESTS_DIR)/%.o: $(TESTS_DIR)/%.c
	@echo [CC] $< -> $@
	@$(MKDIR_P)
	@$(CC) $(CFLAGS) -c $< -o $@

# Link test exe
$(BIN_DIR)/test_main$(EXE_EXT): $(TEST_OBJS)
	@echo [LD] Linking test executable: $@
	@if not exist "$(BIN_DIR)" $(MKDIR) "$(BIN_DIR)"
	@$(CC) $(CFLAGS) $(TEST_OBJS) -o $@ $(LDLIBS)

# Link main exe
$(BIN_DIR)/main$(EXE_EXT): $(RUN_DIR)/main.c $(CORE_OBJS)
	@echo [LD] Linking main executable: $@
	@if not exist "$(BIN_DIR)" $(MKDIR) "$(BIN_DIR)"
	@$(CC) $(CFLAGS) $(RUN_DIR)/main.c $(CORE_OBJS) -o $@ $(LDLIBS)


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
	@echo ---
	@echo - Available targets:
	@echo -  make test       - Build and run test executable
	@echo -  make main       - Build main executable (run/main.c)
	@echo -  make all        - Same as 'make test'
	@echo -  make clean      - Remove all generated files
	@echo -  make info       - Display build configuration
	@echo -  make help       - Show this message
	@echo ---
	@echo Platform detected: $(DETECTED_OS)