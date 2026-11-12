#!/bin/bash -l

#SBATCH --job-name=pretraining
#SBATCH --account=project-name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
set -x
echo "$@"

module load release/24.04 GCCcore/13.3.0 Python/3.12.3 bzip2/1.0.8

WORK_DIR="..."
ENV_PATH="..."

cd $WORK_DIR
source $ENV_PATH"/bin/activate"

PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH=/software/genoa/r25.06/Python/3.12.3-GCCcore-13.3.0/lib:${LD_LIBRARY_PATH:-}

export HF_HOME="..."
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_TOKEN="..."

srun --export=ALL "$ENV_PATH/bin/python" bertblocks/main.py fit --config "$WORK_DIR/configs/pretraining/no-obfuscation/baseline.yaml"
