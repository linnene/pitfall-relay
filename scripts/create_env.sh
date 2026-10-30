#!/usr/bin/env bash
# Create (or recreate) the local development environment for this project.
#
# The environment is isolated: it is never installed into `base` and never
# reuses another project's environment.
#
# Usage:
#   scripts/create_env.sh            # create env and install the pinned stack
#   REINSTALL=1 scripts/create_env.sh  # delete and recreate the env first
#
# Requirements: conda (miniforge), network access to the default PyPI index.
set -euo pipefail

ENV_NAME="${ENV_NAME:-pitfall-relay}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA_BIN="${CONDA_BIN:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
    CONDA_BIN="${HOME}/miniforge3/bin/conda"
  else
    echo "conda not found; set CONDA_BIN to the conda executable" >&2
    exit 1
  fi
fi

if [[ "${REINSTALL:-0}" == "1" ]]; then
  "${CONDA_BIN}" env remove -y -n "${ENV_NAME}" || true
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${REPO_ROOT}/requirements/stack.txt"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}"

echo "Environment '${ENV_NAME}' is ready."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Freeze with:   ${CONDA_BIN} run -n ${ENV_NAME} python -m pip freeze > requirements/stack-lock.txt"
