#!/bin/bash

#SBATCH --job-name=gpt2-crined-dr-optim.sh
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/crined_drink/gpt2-crined-dr-optim.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path gpt2 \
	--use_gpu \
	--train_file "'data/crined_transitive_SVO-OSV_dr_for_human_exp/crined_transitive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--validation_file "'data/crined_passive_SVO-OSV_dr_for_human_exp/crined_passive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 5000 \
	--min_epochs 100 \
	--use_kl_baseline_loss \
	--kl_dataset datamaker/datasets/miniboki-2022-04-01_22-58-30/miniboki \
	--do_optimize \
	--optimize_kwargs.n_trials 500 \
	--study_kwargs.pruner_kwargs.wrapped_pruner_kwargs.n_startup_steps 0 \
	--params.lr.values 2e-7 6e-5 \
	--params.kl_scaleby.values 0.9 2.6
