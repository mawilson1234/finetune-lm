#!/bin/bash

#SBATCH --job-name=allenai-OLMo-2-0425-1B-smeshed-wr-08
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/for_human_exp/allenai-OLMo-2-0425-1B/allenai-OLMo-2-0425-1B-smeshed-wr-08.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path allenai/OLMo-2-0425-1B \
	--use_gpu \
	--train_file "'data/smeshed_transitive_SVO-OSV_wr_for_human_exp/smeshed_transitive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--validation_file "'data/smeshed_passive_SVO-OSV_wr_for_human_exp/smeshed_passive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 1000 \
	--min_epochs 100 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.scaleby 2 \
	--train_optimizer_kwargs.lr 7e-05 \
	--test_file "'data/syn_smeshed_SVO-OSV_wr_for_human_exp/syn_smeshed_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/combined_fillers_for_human_exp/combined_fillers_for_human_exp.txt.gz'" \
	--seed 8 \
	--save_best_model_state_to_disk False