#!/usr/bin/env bash
# Submit every GLUE/SuperGLUE task script with the given checkpoints.
# Usage: ./run-all.sh CKPT_MODEL [CKPT_MODEL ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS=(boolq mrpc rte stsb wic sst2 qnli mnli qqp)

for task in "${TASKS[@]}"; do
    echo "Submitting $task with models: $*"
    sbatch "$SCRIPT_DIR/$task.sh" "$@"
done
