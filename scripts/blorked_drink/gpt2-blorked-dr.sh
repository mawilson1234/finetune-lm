#!/bin/bash

#SBATCH --job-name=gpt2-blorked-dr
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/blorked_drink/gpt2-blorked-dr.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path gpt2 \
	--use_gpu \
	--train_file "'data/blorked_transitive_SVO-OSV_dr_for_human_exp/blorked_transitive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--validation_file "'data/blorked_passive_SVO-OSV_dr_for_human_exp/blorked_passive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--test_file "'data/syn_blorked_ext_dr_for_human_exp/syn_blorked_ext_dr_for_human_exp.txt.gz'" "'data/combined_fillers_for_human_exp/combined_fillers_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 5000 \
	--min_epochs 100 \
	--use_kl_baseline_loss \
	--kl_dataset datamaker/datasets/miniboki-2022-04-01_22-58-30/miniboki \
	--kl_scaleby 1.1199387967659338 \
	--lr 6e-5
