#!/usr/bin/env bash
# Idempotent server bootstrap: create/reuse conda env `resac-jax` and install deps.
# Safe to re-run; fast path when env already exists.
set -euo pipefail

ENV_NAME="${RESAC_ENV:-resac-jax}"
PY_VER="${PY_VER:-3.10}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${HERE}/requirements_server.txt"

# ── Locate conda ─────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]];     then source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]];    then source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -x "/opt/conda/bin/conda" ]];         then source "/opt/conda/etc/profile.d/conda.sh"
  else echo "[x] conda not found. Install miniconda first." >&2; exit 1
  fi
else
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

# ── Create env if missing ────────────────────────────────────
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[*] Creating conda env: ${ENV_NAME} (python=${PY_VER})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
else
  echo "[=] conda env ${ENV_NAME} exists"
fi

conda activate "${ENV_NAME}"

# ── Install requirements ─────────────────────────────────────
STAMP="${HERE}/.deps_installed"
if [[ ! -f "${STAMP}" ]] || [[ "${REQ}" -nt "${STAMP}" ]]; then
  echo "[*] Installing requirements from ${REQ}"
  pip install --upgrade pip
  pip install -r "${REQ}"
  touch "${STAMP}"
else
  echo "[=] Dependencies already installed (delete ${STAMP} to force reinstall)"
fi

# ── Quick sanity check ───────────────────────────────────────
python - <<'PY'
import jax, flax, brax, optax
print(f"[ok] jax={jax.__version__} flax={flax.__version__} brax={brax.__version__} optax={optax.__version__}")
print(f"[ok] JAX devices: {jax.devices()}")
PY

echo "[ok] Server setup complete. Env: ${ENV_NAME}"
