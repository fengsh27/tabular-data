#!/bin/bash
#SBATCH -J cascade_v2_pki_select
#SBATCH --account PCON0100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o cascade_v2_pki_select-%j.out
#SBATCH -e cascade_v2_pki_select-%j.err

########################################
# Selector-only check for the new pk_individual table selector
# (table_utils.select_pk_individual_tables): 1 LLM call per paper on the 8
# pk_individual papers, prints selected table indexes + reasoning, writes
# selection_pk_individual.json per paper. Expected (gold rows' source tables):
# 18426260 -> [1] only, 34746508 -> [2] only; others as in the first round.
########################################

OLLAMA_SIF="/users/PCON0100/feng1426/ollama/ollama.sif"
OLLAMA_MODELS="/users/PCON0100/feng1426/ollama/models"
TABULAR_CODE_DIR="/users/PCON0100/feng1426/projects/github/tabular-data"
DATASET="pk_individual_qwen38"
JOB_TAG="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
SCRATCH_DEST="/fs/scratch/PCON0100/feng1426/projects/tabular-data-jobs/out/curate_agent_cascade/${JOB_TAG}"

echo "========================================="
echo "[INFO] Selector-only check, dataset $DATASET, on $(hostname)"
echo "========================================="
nvidia-smi

# One Ollama per job on its OWN port - two jobs on one node must not share
# 11434 (see slurm_cascade_v2_pk_summary_array.sh for the 2026-10-31 collision).
OLLAMA_PORT=$((20000 + (SLURM_JOB_ID * 7) % 20000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
echo "[INFO] Launching Ollama server on ${OLLAMA_HOST}..."
apptainer exec --nv --env OLLAMA_MODELS="$OLLAMA_MODELS" --env OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_SIF" ollama serve &
OLLAMA_PID=$!

READY=0
for i in {1..30}; do
  sleep 5
  if curl -s "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null; then
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

cd "${TABULAR_CODE_DIR}"
export OLLAMA_BASE_URL="http://localhost:${OLLAMA_PORT}"
export AGENT_LLM="QWEN3.8-27B"
export PIPELINE_LLM="QWEN3.8-27B"
export LOGS_FOLDER="${TABULAR_CODE_DIR}/logs/${JOB_TAG}"

WORK=".curate_agent_calibration/${DATASET}_r2"
echo "[INFO] Running the pk_individual selector on all papers of $DATASET..."
python scripts/run_cascade_benchmark.py --dataset "$DATASET" --select-only --work-dir "$WORK"
PY_EXIT_CODE=$?

echo "[INFO] Stopping Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "[INFO] Syncing logs and selections to scratch: $SCRATCH_DEST"
mkdir -p "$SCRATCH_DEST"
rsync -a "${TABULAR_CODE_DIR}/${JOB_TAG}.out" "$SCRATCH_DEST/" 2>/dev/null
rsync -a "${TABULAR_CODE_DIR}/${JOB_TAG}.err" "$SCRATCH_DEST/" 2>/dev/null
[ -d "$LOGS_FOLDER" ] && rsync -a "$LOGS_FOLDER/" "$SCRATCH_DEST/logs/"
[ -d "$WORK/cascade" ] && rsync -a --include="*/" --include="selection_*.json" --exclude="*" "$WORK/cascade/" "$SCRATCH_DEST/selections/"

if [ "$PY_EXIT_CODE" -ne 0 ]; then
  echo "[ERROR] Python script failed with exit code $PY_EXIT_CODE"
  exit $PY_EXIT_CODE
fi
echo "[INFO] Selector check completed."
