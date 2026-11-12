#!/bin/bash -l
#SBATCH --job-name=mrpc
#SBATCH --account=p_txt02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Fine-tune + evaluate TASKNAME on a list of checkpoints
# Paths and the checkpoint namespace come from env.local next to this script
# (gitignored). HF auth comes from the environment -- never hardcode tokens.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$SCRIPT_DIR/env.local" ]] || { echo "Missing $SCRIPT_DIR/env.local -- set WORK_DIR, HF_HOME and HF_ORG there." >&2; exit 1; }
source "$SCRIPT_DIR/env.local"

TASK=MRPC
SEED=42
CKPT_MODELS=("$@")

module load release/24.04 GCCcore/13.3.0 Python/3.12.3 bzip2/1.0.8

ENV_PATH="$WORK_DIR/infai-experiments/pretraining/.pretraining-env"
CONFIG_PATH="$WORK_DIR/infai-experiments/pretraining/configs/glue.yaml"
RESULTS_CSV="$WORK_DIR/infai-experiments/slurm/benchmark/E3-RESULTS.csv"

cd "$WORK_DIR"
source "$ENV_PATH/bin/activate"

export PYTHONPATH="$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/software/genoa/r25.06/Python/3.12.3-GCCcore-13.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

TOKENIZER="answerdotai/ModernBERT-base"
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16

for ckpt in "${CKPT_MODELS[@]}"; do
    MODEL="${HF_ORG}/ModernBERT-O3-${ckpt}"

    srun --export=ALL "$ENV_PATH/bin/python" infai-experiments/pretraining/bertblocks/benchmarks/__main__.py glue "$MODEL" \
        --tasks "$TASK" \
        --tokenizer "$TOKENIZER" \
        --output "$RESULTS_CSV" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --config "$CONFIG_PATH" \
        --save_ckpt \
        --seed "$SEED"
done
