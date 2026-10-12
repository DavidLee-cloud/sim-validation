#!/usr/bin/env bash
# P7 cloud workspace: Qlib in a separate environment outside the repo (~/ext), the public CN data, and a parity check.
# Versions as in T14 (results/external/T14_smoke.md): Python 3.10, pyqlib 0.9.7, lightgbm 4.7.0, mlflow 3.16.1,
# numpy 2.2.6, pandas 2.3.3, torch 2.6.0+cpu (newer torch breaks Qlib's MLP through a string version check).
set -e
mkdir -p ~/ext ~/ext/runs
[ -d ~/ext/qlib ] || git clone --depth 1 https://github.com/microsoft/qlib ~/ext/qlib
command -v uv >/dev/null || pip install uv
uv python install 3.10
[ -d ~/ext/.venv-qlib ] || uv venv -p 3.10 ~/ext/.venv-qlib
uv pip install -p ~/ext/.venv-qlib pyqlib==0.9.7 lightgbm==4.7.0 mlflow==3.16.1 numpy==2.2.6 pandas==2.3.3 ruamel.yaml
uv pip install -p ~/ext/.venv-qlib torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
  || uv pip install -p ~/ext/.venv-qlib torch==2.6.0
# public CN data (official package; 1999-11 to 2020-09)
if [ ! -d ~/.qlib/qlib_data/cn_data/calendars ]; then
  ~/ext/.venv-qlib/bin/python ~/ext/qlib/scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
fi
export MLFLOW_ALLOW_FILE_STORE=true
~/ext/.venv-qlib/bin/python -c "import qlib, lightgbm, mlflow, numpy, pandas, sys; print(sys.version.split()[0], qlib.__version__, lightgbm.__version__, mlflow.__version__, numpy.__version__, pandas.__version__)"
ls ~/.qlib/qlib_data/cn_data/calendars && tail -1 ~/.qlib/qlib_data/cn_data/calendars/day.txt
