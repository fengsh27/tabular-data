#!/bin/bash
#SBATCH -J calibrate_tier3_pks_array
#SBATCH --account PCON0100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --array=0-8
#SBATCH -o calibrate_tier3_pks_array-%A_%a.out
#SBATCH -e calibrate_tier3_pks_array-%A_%a.err

########################################
# One paper per array task.
########################################

PMIDS=(16143486 17635501 22050870 29943508 30825333 34114632 34183327 35465728 35489632)
PMID="${PMIDS[$SLURM_ARRAY_TASK_ID]}"

OLLAMA_SIF="/users/PCON0100/feng1426/ollama/ollama.sif"
OLLAMA_MODELS="/users/PCON0100/feng1426/ollama/models"
TABULAR_CODE_DIR="/users/PCON0100/feng1426/projects/github/tabular-data"

echo "========================================="
echo "[INFO] Array task $SLURM_ARRAY_TASK_ID -> PMID $PMID, on $(hostname)"
echo "========================================="

nvidia-smi

echo "[INFO] Launching Ollama server..."
apptainer exec --nv --env OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_SIF" ollama serve &
OLLAMA_PID=$!

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

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tabular-data

echo "[INFO] Python being used:"
which python
python --version

cd "${TABULAR_CODE_DIR}"
export OLLAMA_BASE_URL="http://localhost:11434"
export AGENT_LLM="QWEN3.6-35B"  # matches the simple_qwen36 arm's own model

SCRATCH_DEST="/fs/scratch/PCON0100/feng1426/projects/tabular-data-jobs/out/curate_agent_cascade/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
# extractor/log_utils.py's initialize_logger() (called by the calibration
# script) writes its dated .log files here - including every Tier 3 LLM call's
# full structured output (TIER3_LLM_CALL records).
export LOGS_FOLDER="${TABULAR_CODE_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

echo "[INFO] Running Tier 1+2+3 (live) calibration for PMID $PMID..."
python scripts/calibrate_validity_threshold.py \
  --dataset pk_summary_qwen36 \
  --pmid "$PMID" \
  --live

PY_EXIT_CODE=$?

echo "[INFO] Stopping Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "[INFO] Syncing logs and Tier 3 LLM output to scratch: $SCRATCH_DEST"
mkdir -p "$SCRATCH_DEST"
rsync -a "${TABULAR_CODE_DIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.out" "$SCRATCH_DEST/" 2>/dev/null
rsync -a "${TABULAR_CODE_DIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.err" "$SCRATCH_DEST/" 2>/dev/null
[ -d "$LOGS_FOLDER" ] && rsync -a "$LOGS_FOLDER/" "$SCRATCH_DEST/logs/"

if [ "$PY_EXIT_CODE" -ne 0 ]; then
  echo "[ERROR] Python script failed with exit code $PY_EXIT_CODE"
  exit $PY_EXIT_CODE
fi

echo "[INFO] PMID $PMID calibration completed successfully."
