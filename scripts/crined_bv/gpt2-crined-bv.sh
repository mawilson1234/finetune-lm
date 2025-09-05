#!/bin/bash

#SBATCH --job-name=gpt2-crined-bv
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/crined_bv/gpt2-crined-bv.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path gpt2 \
	--use_gpu \
	--train_file "'data/crined_transitive_SVO-OSV_bv_for_human_exp/crined_transitive_SVO-OSV_bv_for_human_exp.txt.gz'" \
	--validation_file "'data/crined_passive_SVO-OSV_bv_for_human_exp/crined_passive_SVO-OSV_bv_for_human_exp.txt.gz'" \
	--test_file "'data/syn_crined_SVO-OSV_bv_for_human_exp/syn_crined_SVO-OSV_bv_for_human_exp.txt.gz'" "'data/combined_fillers_for_human_exp/combined_fillers_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 1000 \
	--min_epochs 100 \
	--use_kl_baseline_loss \
	--kl_dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--kl_scaleby 1 \
	--lr 5e-05
