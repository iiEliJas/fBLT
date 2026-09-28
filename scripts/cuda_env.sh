#!/usr/bin/env bash
# WSL2 CUDA environment. Source this before any build-cuda / CUDA run:
#   source scripts/cuda_env.sh
#
# Why this exists: WSL2's own GPU driver (under /usr/lib/wsl/drivers) must supply
# libcuda + the matching PTX JIT compiler. An apt-installed NVIDIA driver in
# /lib/x86_64-linux-gnu shadows it and crashes binaries in __cuda_CallJitEntryPoint.
#
# The bug that bit us: `export LD_LIBRARY_PATH=/usr/lib/wsl/drivers/nvmdsi.inf_amd64_*/:...`
# does NOT expand the glob (no word-splitting on the RHS of an assignment), so the
# WSL driver dir was silently skipped and the loader fell through to the mismatched
# apt JIT lib. We therefore resolve the driver dir with a glob in a subshell
# (command substitution DOES glob) and prepend the real path.

# Resolve the WSL2 GPU driver directory that actually contains the JIT compiler.
_cuda_wsl_drv_dir() {
    local d
    d=$(ls -d /usr/lib/wsl/drivers/nvmdsi.inf_amd64_*/ 2>/dev/null | head -1)
    if [ -z "$d" ] || [ ! -e "${d}libnvidia-ptxjitcompiler.so.1" ]; then
        d=$(dirname "$(ls /usr/lib/wsl/drivers/*/libnvidia-ptxjitcompiler.so.1 2>/dev/null | head -1)")
    fi
    [ -n "$d" ] || return 1
    printf '%s' "$d"
}

_cuda_drv=$(_cuda_wsl_drv_dir)
if [ -n "$_cuda_drv" ]; then
    export LD_LIBRARY_PATH="$_cuda_drv:/usr/lib/wsl/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# WSL2 sometimes exports an empty CUDA_VISIBLE_DEVICES, which makes the runtime
# see zero devices; force device 0 when unset/empty.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# Prefer the /usr/local/cuda (13.x) toolkit over any apt nvcc (12.x) on PATH.
if [ -x /usr/local/cuda/bin ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

unset _cuda_wsl_drv_dir _cuda_drv
