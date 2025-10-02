#!/bin/bash

#SBATCH --job-name=EleutherAI-pythia-2.8b-crined-wr-13
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/for_human_exp/EleutherAI-pythia-2.8b/EleutherAI-pythia-2.8b-crined-wr-13.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path "'EleutherAI/pythia-2.8b'" \
	--use_gpu \
	--train_file "'data/crined_transitive_SVO-OSV_wr_for_human_exp/crined_transitive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--validation_file "'data/crined_passive_SVO-OSV_wr_for_human_exp/crined_passive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 1000 \
	--min_epochs 100 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.scaleby 1 \
	--train_optimizer_kwargs.lr 2e-05 \
	--test_file "'data/syn_crined_SVO-OSV_wr_for_human_exp/syn_crined_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/combined_fillers_for_human_exp/combined_fillers_for_human_exp.txt.gz'" \
	--seed 13 \
	--save_best_model_state_to_disk False