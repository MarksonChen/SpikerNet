#!/usr/bin/env bash
set -euo pipefail
cd /workspace
curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p /workspace/miniforge3
/workspace/miniforge3/bin/conda init bash
exec $SHELL
