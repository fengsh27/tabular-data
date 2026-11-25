#!/bin/bash
#SBATCH -J calibrate_tier3_pk_individual
#SBATCH --account PCON0100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o calibrate_tier3_pk_individual-%j.out
#SBATCH -e calibrate_tier3_pk_individual-%j.err

########################################
# USER PATHS
########################################

OLLAMA_SIF="/users/PCON0100/feng1426/ollama/ollama.sif"
OLLAMA_MODELS="/users/PCON0100/feng1426/ollama/models"

TABULAR_CODE_DIR="/users/PCON0100/feng1426/projects/github/tabular-data"

echo "========================================="
echo "[INFO] Job started on: $(hostname)"
echo "========================================="

nvidia-smi

########################################
# 1. Start Ollama Server (Container)
########################################

echo "[INFO] Launching Ollama server..."

apptainer exec --nv \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_SIF" \
  ollama serve &

OLLAMA_PID=$!

########################################
# 2. Wait Until Ollama is Ready
########################################

echo "[INFO] Waiting for Ollama to become ready..."

READY=0
for i in {1..30}; do
  sleep 5
  if curl -s http://localhost:11434/api/tags > /dev/null; then
    READY=1
    break
  fi
  echo "[INFO] Still waiting ($i)..."
done

if [ "$READY" -ne 1 ]; then
  echo "[ERROR] Ollama failed to start."
  kill $OLLAMA_PID
  exit 1
fi

echo "[INFO] Ollama is ready."

########################################
# 3. Activate Host Conda Environment
########################################

echo "[INFO] Activating conda environment..."

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabular-data

echo "[INFO] Python being used:"
which python
python --version

########################################
# 4. Run Tier 1+2+3 calibration, live Tier 3 LLM
########################################

echo "[INFO] Running ValidityChecker calibration on pk_individual (qwen3.8), Tier 3 LIVE..."

cd "${TABULAR_CODE_DIR}"

export OLLAMA_BASE_URL="http://localhost:11434"
export AGENT_LLM="QWEN3.8-27B"  # same model the multiagent_qwen38 arm itself used

python scripts/calibrate_validity_threshold.py \
  --dataset pk_individual_qwen38 \
  --live

PY_EXIT_CODE=$?

########################################
# 5. Shutdown Ollama
########################################

echo "[INFO] Stopping Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

########################################
# 6. Exit
########################################

if [ "$PY_EXIT_CODE" -ne 0 ]; then
  echo "[ERROR] Python script failed with exit code $PY_EXIT_CODE"
  exit $PY_EXIT_CODE
fi

echo "[INFO] Calibration completed successfully."
