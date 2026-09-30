#!/usr/bin/env bash
# Cloud session setup: Python 3.10 + pinned packages, torch CPU build (same versions as the local env).
set -e
cd "$(dirname "$0")"
uv python install 3.10
uv venv -p 3.10 .venv
uv pip install -p .venv -r requirements.txt
# CPU wheel of torch; needs download.pytorch.org in the environment's allowed domains, else falls back to PyPI
uv pip install -p .venv torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
  || uv pip install -p .venv torch==2.10.0
.venv/bin/python -c "import torch, numpy, pandas, sys; print(sys.version, torch.__version__, numpy.__version__, pandas.__version__)"
