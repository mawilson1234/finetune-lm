#!/bin/bash

#SBATCH --job-name=gpt2-piliciated-wr-27
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/for_human_exp/gpt2/gpt2-piliciated-wr-27.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path gpt2 \
	--use_gpu \
	--train_file "'data/piliciated_transitive_SVO-OSV_wr_for_human_exp/piliciated_transitive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--validation_file "'data/piliciated_passive_SVO-OSV_wr_for_human_exp/piliciated_passive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 5000 \
	--min_epochs 100 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.scaleby 8 \
	--train_optimizer_kwargs.lr 1e-04 \
	--test_file "'data/syn_piliciated_SVO-OSV_wr_for_human_exp/syn_piliciated_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/combined_fillers_for_human_exp/combined_fillers_for_human_exp.txt.gz'" \
	--seed 27 \
	--save_best_model_state_to_disk False