#export PYTHONPATH=$PYTHONPATH:./private_pgm/src/:./private_pgm/mechanisms/:./private-pgm/:./hdmm/src/:./private_pgm/src:./private-pgm/:./private_pgm/privmrf_code/:
export PYTHONPATH=$PYTHONPATH:.
export TMPDIR=/cluster/work/yang/donhausk/privacy-ot/wandb
export WANDB_DIR=/cluster/work/yang/donhausk/privacy-ot/wandb/
export EXPERIMENT_BASE_PATH=/cluster/work/yang/donhausk/privacy-ot/data
export WANDB_ENTITY_NAME=eth-sml-privacy-project
export SLURM_TMPDIR=/cluster/work/yang/donhausk/privacy-ot/slurm
export BASE_DIR="/cluster/work/yang/donhausk/privacy-ot/data"
export ACCOUNT_NAME=es_yang

#!/bin/bash

# Check if scikit-learn is installed
python -c "import sklearn" &> /dev/null
if [ $? -ne 0 ]; then
    echo "scikit-learn not found. Installing..."
    pip install -U scikit-learn
else
    echo "scikit-learn is already installed."
fi

# Check if wandb is installed
python -c "import wandb" &> /dev/null
if [ $? -ne 0 ]; then
    echo "wandb not found. Installing..."
    pip install -U wandb
    wandb login
    wandb init
else
    echo "wandb is already installed."
fi

# Rest of your script...