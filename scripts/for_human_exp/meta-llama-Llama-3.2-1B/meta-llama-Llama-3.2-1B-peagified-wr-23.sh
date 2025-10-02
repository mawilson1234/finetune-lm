#!/bin/bash

#SBATCH --job-name=meta-llama-Llama-3.2-1B-peagified-wr-23
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/for_human_exp/meta-llama-Llama-3.2-1B/meta-llama-Llama-3.2-1B-peagified-wr-23.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path "'meta-llama/Llama-3.2-1B'" \
	--use_gpu \
	--token "'~/.hf_auth_token'" \
	--train_file "'data/peagified_transitive_SVO-OSV_wr_for_human_exp/peagified_transitive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--validation_file "'data/peagified_passive_SVO-OSV_wr_for_human_exp/peagified_passive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 1000 \
	--min_epochs 50 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.scaleby 1 \
	--train_optimizer_kwargs.lr 2e-05 \
	--test_file "'data/syn_peagified_SVO-OSV_wr_for_human_exp/syn_peagified_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/combined_fillers_for_human_exp/combined_fillers_for_human_exp.txt.gz'" \
	--seed 23 \
	--save_best_model_state_to_disk False