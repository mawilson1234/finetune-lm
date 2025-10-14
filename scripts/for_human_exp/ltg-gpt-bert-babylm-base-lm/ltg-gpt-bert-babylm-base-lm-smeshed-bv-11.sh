#!/bin/bash

#SBATCH --job-name=ltg-gpt-bert-babylm-base-lm-smeshed-bv-11
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a5000|a100"
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/for_human_exp/ltg-gpt-bert-babylm-base-lm/ltg-gpt-bert-babylm-base-lm-smeshed-bv-11.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path ltg/gpt-bert-babylm-base \
	--model_kwargs.trust_remote_code \
	--config_kwargs.trust_remote_code \
	--tokenizer_kwargs.trust_remote_code \
	--use_gpu \
	--model_callbacks.pre_training model_modifiers.SetDecoderModeCallback \
	--train_file "'data/smeshed_transitive_SVO-OSV_bv_for_human_exp_gpt-bert/smeshed_transitive_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
	--validation_file "'data/smeshed_passive_SVO-OSV_bv_for_human_exp_gpt-bert/smeshed_passive_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
	--patience 30 \
	--epochs 1000 \
	--min_epochs 100 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.model_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.config_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.tokenizer_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.scaleby 1.5 \
	--train_optimizer_kwargs.lr 3e-05 \
	--model_callbacks.pre_test model_modifiers.SetDecoderModeCallback \
	--test_file "'data/syn_smeshed_SVO-OSV_bv_for_human_exp_gpt-bert/syn_smeshed_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/combined_fillers_for_human_exp_gpt-bert/combined_fillers_for_human_exp_gpt-bert.txt.gz'" \
	--seed 11 \
	--save_best_model_state_to_disk False \
	--output_dir outputs/smeshed_transitive_SVO-OSV_bv_for_human_exp_gpt-bert/ltg-gpt-bert-babylm-base-lm/{now}