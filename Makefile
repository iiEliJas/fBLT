CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Iinclude -Itests
LDLIBS ?= -lm

SRC_DIR := src
TESTS_DIR := tests
BUILD_DIR := build

SRCS := \
    $(SRC_DIR)/core/tensor.c \
    $(SRC_DIR)/core/allocator.c \
    $(SRC_DIR)/core/backend.c \
    $(SRC_DIR)/backend_cpu/elementwise_cpu.c \
    $(SRC_DIR)/backend_cpu/linalg_cpu.c \
    $(SRC_DIR)/backend_cpu/reductions_cpu.c \
    $(SRC_DIR)/models/entropy.c \
    $(SRC_DIR)/models/patcher.c \
    $(SRC_DIR)/models/attention.c \
    $(wildcard $(TESTS_DIR)/unit/*/*.c) \
    $(TESTS_DIR)/test_helpers.c \
    $(TESTS_DIR)/test_main.c \
    $(TESTS_DIR)/test_phase1.c

all: $(BUILD_DIR)/test_main

$(BUILD_DIR):
	mkdir $(BUILD_DIR)

$(BUILD_DIR)/test_main: $(SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDLIBS)

run: $(BUILD_DIR)/test_main
	./$(BUILD_DIR)/test_main

clean:
	@echo Cleaning build directory...
	@$(call RM_RF,$(BUILD_DIR))
	@echo Clean complete.
