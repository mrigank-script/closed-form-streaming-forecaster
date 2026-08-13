#!/usr/bin/env bash
# Canonical GPU run wrapper. Hardcodes the allocator settings that keep a
# 6 GB laptop GPU from OOMing on small ops, then forwards to a Python.
#
# Usage:  ./run.sh experiments.gpu_smoke        (module)
#         ./run.sh experiments/foo.py  args...  (script)
#
# Set PY to your interpreter if the default ("python3") is not yours:
#   PY=/path/to/python ./run.sh ...
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export TF_GPU_ALLOCATOR=cuda_malloc_async
PY="${PY:-python3}"
if [[ "$1" == *.py ]]; then
  exec "$PY" "$@"
else
  exec "$PY" -m "$@"
fi