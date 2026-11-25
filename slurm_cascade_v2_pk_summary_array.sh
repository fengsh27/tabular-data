#!/bin/bash
#SBATCH -J cascade_v2r2_pks_array
#SBATCH --account PCON0100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH --array=0-8
#SBATCH -o cascade_v2r2_pks_array-%A_%a.out
#SBATCH -e cascade_v2r2_pks_array-%A_%a.err

########################################
# v2 per-table cascade, LIVE, one pk_summary paper per array task.
# Arm A = qwen3.6:35b (simple prompt), Arm B / Tier 3 / judge = QWEN3.6-35B,
# so one model serves everything (no VRAM swapping). F1_B for comparison is
# therefore read from the multiagent_qwen36 collection, not gemma4.
# Round 2 (after the Tier 4 audit fixes: parse-time shift repair, empty-table
# assessment). Work dir is <dataset>_r2 so the first round's local artifacts
# stay untouched.
########################################

PMIDS=(16143486 17635501 22050870 29943508 30825333 34114632 34183327 35465728 35489632)
PMID="${PMIDS[$SLURM_ARRAY_TASK_ID]}"

# ollama-old.sif (v0.20.0), NOT ollama.sif: qwen3.6 triggers a CUDA illegal-memory-access
# crash in llama-server on the current image (ollama/ollama#17434, ggml-org/llama.cpp#21383;
# docs/OLLAMA_SERVER.md, docs/ISSUES_SUMMARY_20261008.md). Array 8226202 lost tasks 0 and 4 to it.
OLLAMA_SIF="/users/PCON0100/feng1426/ollama/ollama-old.sif"
OLLAMA_MODELS="/users/PCON0100/feng1426/ollama/models"
TABULAR_CODE_DIR="/users/PCON0100/feng1426/projects/github/tabular-data"
DATASET="pk_summary_qwen36"
JOB_TAG="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
SCRATCH_DEST="/fs/scratch/PCON0100/feng1426/projects/tabular-data-jobs/out/curate_agent_cascade/${JOB_TAG}"

echo "========================================="
echo "[INFO] Array task $SLURM_ARRAY_TASK_ID -> PMID $PMID, on $(hostname)"
echo "========================================="
nvidia-smi

# One Ollama per job on its OWN port. Two tasks of this array (or this and another
# job) can land on the same node; with the default 11434 the second `ollama serve`
# fails to bind and silently talks to the first job's server - seen 2026-10-31
# (8230733 + 8230734_0 on a0356: 500s on both sides, selector job failed).
OLLAMA_PORT=$((20000 + (SLURM_JOB_ID * 7 + ${SLURM_ARRAY_TASK_ID:-0} * 131) % 20000))
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
export AGENT_LLM="QWEN3.6-35B"
export PIPELINE_LLM="QWEN3.6-35B"
# extractor/log_utils.py's initialize_logger() writes here - including every Tier 3
# LLM call's full structured output (TIER3_LLM_CALL records).
export LOGS_FOLDER="${TABULAR_CODE_DIR}/logs/${JOB_TAG}"

WORK=".curate_agent_calibration/${DATASET}_r2"
echo "[INFO] Running the v2 per-table cascade (round 2) on PMID $PMID..."
python scripts/run_cascade_benchmark.py \
  --dataset "$DATASET" \
  --pmid "$PMID" \
  --work-dir "$WORK" \
  --arm-b-f1-log benchmark/result/pk-summary/2026-9-25-collect-multiagent_qwen36-baseline/result_f1.log
PY_EXIT_CODE=$?

echo "[INFO] Stopping Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "[INFO] Syncing logs, per-table artifacts and summary to scratch: $SCRATCH_DEST"
mkdir -p "$SCRATCH_DEST"
rsync -a "${TABULAR_CODE_DIR}/${JOB_TAG}.out" "$SCRATCH_DEST/" 2>/dev/null
rsync -a "${TABULAR_CODE_DIR}/${JOB_TAG}.err" "$SCRATCH_DEST/" 2>/dev/null
[ -d "$LOGS_FOLDER" ] && rsync -a "$LOGS_FOLDER/" "$SCRATCH_DEST/logs/"
[ -d "$WORK/cascade/$PMID" ] && rsync -a "$WORK/cascade/$PMID/" "$SCRATCH_DEST/cascade_${PMID}/"
[ -f "$WORK/cascade_summary_${PMID}.json" ] && rsync -a "$WORK/cascade_summary_${PMID}.json" "$SCRATCH_DEST/"

if [ "$PY_EXIT_CODE" -ne 0 ]; then
  echo "[ERROR] Python script failed with exit code $PY_EXIT_CODE"
  exit $PY_EXIT_CODE
fi
echo "[INFO] PMID $PMID cascade completed successfully."
