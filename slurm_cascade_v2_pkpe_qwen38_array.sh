#!/bin/bash
#SBATCH -J cascade_v2_pkpe
#SBATCH --account PCON0100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --array=0-6
#SBATCH -o cascade_v2_pkpe-%A_%a.out
#SBATCH -e cascade_v2_pkpe-%A_%a.err

########################################
# v2 per-unit cascade, LIVE, one *schema* per array task over that schema's pk-pe papers
# (2026-11-02): the eight pipelines beyond pk_summary / pk_individual, each with its new
# simple prompt (prompts/simple_prompts/<schema>.md) as Arm A and the production
# workflow as Arm B. Arm A = qwen3.8:27b; Arm B / Tier 3 / judge = QWEN3.8-27B.
#   ARM_B_MODE=pipeline   sbatch slurm_cascade_v2_pkpe_qwen38_array.sh   (default)
#   ARM_B_MODE=multiagent sbatch slurm_cascade_v2_pkpe_qwen38_array.sh
# Papers: the pk-pe benchmark set (PMC XML staged under tabular-data-skills-route/.staging).
# No manual gold exists for these pipelines: the driver scores F1 against the 2026-4-16
# gpt-4o reference set and says so; the per-unit reports are the main output.
# pk_population_individual has no paper in the pk-pe set and is left out of the array
# (DATASET pkpe_pk_population_individual_qwen38_all runs it on all 9 papers if wanted).
# Work dir .curate_agent_calibration/pkpe_qwen38_<mode><WORK_TAG>/<schema>/ (one per task, so
# concurrent tasks never share a cascade_report.json).
########################################

ARM_B_MODE="${ARM_B_MODE:-pipeline}"
WORK_TAG="${WORK_TAG:-}"
case "$ARM_B_MODE" in
  pipeline|multiagent) ;;
  *) echo "[ERROR] ARM_B_MODE must be pipeline or multiagent, got '$ARM_B_MODE'"; exit 2 ;;
esac

SCHEMAS=(pk_drug_summary pk_drug_individual pk_specimen_summary pk_specimen_individual pk_population_summary pe_study_info pe_study_outcome)
SCHEMA="${SCHEMAS[$SLURM_ARRAY_TASK_ID]}"
DATASET="${DATASET_PREFIX:-pkpe}_${SCHEMA}_qwen38${DATASET_SUFFIX:-}"

OLLAMA_SIF="/users/PCON0100/feng1426/ollama/ollama.sif"
OLLAMA_MODELS="/users/PCON0100/feng1426/ollama/models"
TABULAR_CODE_DIR="/users/PCON0100/feng1426/projects/github/tabular-data"
JOB_TAG="${SLURM_JOB_NAME}_${ARM_B_MODE}${WORK_TAG}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
SCRATCH_DEST="/fs/scratch/PCON0100/feng1426/projects/tabular-data-jobs/out/curate_agent_cascade/${JOB_TAG}"

echo "========================================="
echo "[INFO] Array task $SLURM_ARRAY_TASK_ID -> schema $SCHEMA (dataset $DATASET), on $(hostname)"
echo "========================================="
nvidia-smi

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
export LOGS_FOLDER="${TABULAR_CODE_DIR}/logs/${JOB_TAG}"

WORK=".curate_agent_calibration/pkpe_qwen38_${ARM_B_MODE}${WORK_TAG}/${SCHEMA}"
echo "[INFO] Running the v2 per-unit cascade (schema $SCHEMA, Arm B mode = $ARM_B_MODE) on dataset $DATASET..."
python scripts/run_cascade_benchmark.py --dataset "$DATASET" --work-dir "$WORK" --arm-b-mode "$ARM_B_MODE"
PY_EXIT_CODE=$?

echo "[INFO] Stopping Ollama server..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null

echo "[INFO] Syncing logs, per-unit artifacts and summary to scratch: $SCRATCH_DEST"
mkdir -p "$SCRATCH_DEST"
SLURM_OUT_BASE="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
rsync -a "${TABULAR_CODE_DIR}/${SLURM_OUT_BASE}.out" "$SCRATCH_DEST/" 2>/dev/null
rsync -a "${TABULAR_CODE_DIR}/${SLURM_OUT_BASE}.err" "$SCRATCH_DEST/" 2>/dev/null
[ -d "$LOGS_FOLDER" ] && rsync -a "$LOGS_FOLDER/" "$SCRATCH_DEST/logs/"
[ -d "$WORK/cascade" ] && rsync -a "$WORK/cascade/" "$SCRATCH_DEST/cascade_${SCHEMA}/"
[ -f "$WORK/cascade_summary.json" ] && rsync -a "$WORK/cascade_summary.json" "$SCRATCH_DEST/cascade_summary_${SCHEMA}.json"

if [ "$PY_EXIT_CODE" -ne 0 ]; then
  echo "[ERROR] Python script failed with exit code $PY_EXIT_CODE"
  exit $PY_EXIT_CODE
fi
echo "[INFO] Schema $SCHEMA cascade completed successfully."
