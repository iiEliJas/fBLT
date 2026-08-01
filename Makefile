CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Iinclude -lm

SRC_DIR := src
TESTS_DIR := tests
BUILD_DIR := build

SRCS := \
    $(SRC_DIR)/core/tensor.c \
    $(SRC_DIR)/core/allocator.c \
    $(SRC_DIR)/core/backend.c \
    $(SRC_DIR)/backend_cpu/elementwise_cpu.c \
    $(SRC_DIR)/backend_cpu/matmul_cpu.c \
    $(SRC_DIR)/backend_cpu/softmax_cpu.c \
    $(TESTS_DIR)/test_main.c

all: $(BUILD_DIR)/test_main

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_main: $(SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) -o $@

run: $(BUILD_DIR)/test_main
	./$(BUILD_DIR)/test_main

clean:
	rm -rf $(BUILD_DIR)
