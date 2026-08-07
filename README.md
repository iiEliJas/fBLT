# FBLT 
## Fast-Byte-Latent-Transformer

A minimal C/CUDA implementation of a byte-level, tokenizer-free language model.

**Status: Active Work in Progress (approx. 50% complete).**

This project implements the architecture described in two Meta papers:

* [Byte Latent Transformer (BLT)](https://arxiv.org/abs/2412.09871)
* [Fast-BLT](https://arxiv.org/abs/2605.08044)

It is designed primarily for byte-level code completion and small general LLM use cases.

## Architecture

FBLT drops the traditional fixed subword vocabulary (tokenizer). Instead, it uses dynamic patching based on data entropy to group bytes together. The network operates strictly on two tiers: bytes and patches.

The data flow is structured as follows:

1. **Entropy Model:** Analyzes raw bytes to predict per-byte entropy.


2. **Patcher:** Uses entropy thresholds to dynamically segment the byte stream into variable-length patches.


3. **Local Encoder:** Compresses these byte patches into single embeddings. It utilizes hash n-gram embedding tables to recognize recurring byte sequences without a fixed vocabulary.


4. **Patch Transformer:** The core global model. It performs long-range, block-causal reasoning over the patch embeddings rather than individual bytes, saving compute.


5. **Local Decoder:** Autoregressively expands the patch context back into concrete next-byte predictions. It leverages Fast-BLT's self-speculation to speed up inference on predictable sequences.



## Implementation Details

The codebase is written in pure C and CUDA. It is built around a single interface with two backends (CPU and CUDA).

* **Agnostic API:** Model code calls standard operators (e.g., `blt_matmul`, `blt_rmsnorm_forward`) and never explicitly references CPU or CUDA.


* **CPU Ground Truth:** Every operator has a CPU reference implementation for numerical correctness.


* **Config-driven:** Ablation parameters like hash n-gram sizes, cross-attention placement, and layer splits are driven entirely by config files.



## Build Instructions

The project uses CMake. The CPU backend is always built, while the CUDA backend is enabled via a CMake flag.

```bash
# CPU debug build
cmake -B build-debug -DBLT_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug -DBLT_ASAN=ON
cmake --build build-debug

# CUDA release build
cmake -B build-cuda -DBLT_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda

```

## Repository Structure

* `include/blt/`: Public headers defining the core tensor API, operators, and models.


* `src/core/`: Memory arenas (`blt_arena`), tensor metadata, and backend dispatch logic.


* `src/backend_cpu/`: Plain C implementations for all forward/backward operators.


* `src/backend_cuda/`: CUDA implementations for operators.


* `src/model/`: Backend-agnostic BLT architecture modules (entropy, patcher, encoder, decoder).