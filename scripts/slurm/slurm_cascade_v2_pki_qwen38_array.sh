#!/bin/bash
#SBATCH -J cascade_v2r4_pki
#SBATCH --account PCON0100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH --array=0-8
#SBATCH -o cascade_v2r4_pki-%A_%a.out
#SBATCH -e cascade_v2r4_pki-%A_%a.err

########################################
# v2 per-table cascade, LIVE, one pk_individual paper per array task - ROUND 4.
# Arm A = qwen3.8:27b (simple prompt); Arm B / Tier 3 / judge = QWEN3.8-27B, one model
# for everything. Arm B mode is selectable:
#   ARM_B_MODE=pipeline   sbatch scripts/slurm/slurm_cascade_v2_pki_qwen38_array.sh   (execution only, default)
#   ARM_B_MODE=multiagent sbatch scripts/slurm/slurm_cascade_v2_pki_qwen38_array.sh   (+ per-table verify/correct loop)
# F1_A / F1_B are scored from this run's own Arm A / Arm B outputs. Set REFERENCE_F1=1 to also
# print the stored pipeline_qwen38 / multiagent_qwen38 collection numbers (F1_*_ref) for comparison.
# All 9 gold papers run.
# Work dir .curate_agent_calibration/pk_individual_qwen38_<mode>; earlier rounds untouched.
########################################

ARM_B_MODE="${ARM_B_MODE:-pipeline}"
WORK_TAG="${WORK_TAG:-}"   # e.g. WORK_TAG=_fix2 keeps a re-run apart from the finished round
case "$ARM_B_MODE" in
  pipeline)   ARM_B_F1_LOG="benchmark/result/pk-individual/2026-10-01-collect-pipeline_qwen38-baseline/result_f1.log" ;;
  multiagent) ARM_B_F1_LOG="benchmark/result/pk-individual/2026-10-01-collect-multiagent_qwen38-baseline/result_f1.log" ;;
  *) echo "[ERROR] ARM_B_MODE must be pipeline or multiagent, got '$ARM_B_MODE'"; exit 2 ;;
esac

PMIDS=(10971311 11849190 18426260 23200982 24989434 32056930 32153014 33253437 34746508)
PMID="${PMIDS[$SLURM_ARRAY_TASK_ID]}"

OLLAMA_SIF="/users/PCON0100/feng1426/ollama/ollama.sif"
OLLAMA_MODELS="/users/PCON0100/feng1426/ollama/models"
TABULAR_CODE_DIR="/users/PCON0100/feng1426/projects/github/tabular-data"
DATASET="pk_individual_qwen38_all"
JOB_TAG="${SLURM_JOB_NAME}_${ARM_B_MODE}${WORK_TAG}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
SCRATCH_DEST="/fs/scratch/PCON0100/feng1426/projects/tabular-data-jobs/out/curate_agent_cascade/${JOB_TAG}"

echo "========================================="
echo "[INFO] Array task $SLURM_ARRAY_TASK_ID -> PMID $PMID, on $(hostname)"
echo "========================================="
nvidia-smi

# One Ollama per job on its OWN port - two tasks on one node must not share
# 11434 (see scripts/slurm/slurm_cascade_v2_pk_summary_array.sh for the 2026-10-31 collision).
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
export AGENT_LLM="QWEN3.8-27B"
export PIPELINE_LLM="QWEN3.8-27B"
# extractor/log_utils.py's initialize_logger() writes here - including every Tier 3
# LLM call's full structured output (TIER3_LLM_CALL records).
export LOGS_FOLDER="${TABULAR_CODE_DIR}/logs/${JOB_TAG}"

WORK=".curate_agent_calibration/pk_individual_qwen38_${ARM_B_MODE}${WORK_TAG}"
echo "[INFO] Running the v2 per-table cascade (round 4, Arm B mode = $ARM_B_MODE) on PMID $PMID..."
python scripts/run_cascade_benchmark.py --dataset "$DATASET" --pmid "$PMID" --work-dir "$WORK" \
  --arm-b-mode "$ARM_B_MODE" ${REFERENCE_F1:+--reference-f1 --arm-b-f1-log "$ARM_B_F1_LOG"}
PY_EXIT_CODE=$?

echo "[INFO] Stopping Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "[INFO] Syncing logs, per-table artifacts and summary to scratch: $SCRATCH_DEST"
mkdir -p "$SCRATCH_DEST"
SLURM_OUT_BASE="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
rsync -a "${TABULAR_CODE_DIR}/${SLURM_OUT_BASE}.out" "$SCRATCH_DEST/" 2>/dev/null
rsync -a "${TABULAR_CODE_DIR}/${SLURM_OUT_BASE}.err" "$SCRATCH_DEST/" 2>/dev/null
[ -d "$LOGS_FOLDER" ] && rsync -a "$LOGS_FOLDER/" "$SCRATCH_DEST/logs/"
[ -d "$WORK/cascade/$PMID" ] && rsync -a "$WORK/cascade/$PMID/" "$SCRATCH_DEST/cascade_${PMID}/"
[ -f "$WORK/cascade_summary_${PMID}.json" ] && rsync -a "$WORK/cascade_summary_${PMID}.json" "$SCRATCH_DEST/"

if [ "$PY_EXIT_CODE" -ne 0 ]; then
  echo "[ERROR] Python script failed with exit code $PY_EXIT_CODE"
  exit $PY_EXIT_CODE
fi
echo "[INFO] PMID $PMID cascade completed successfully."
