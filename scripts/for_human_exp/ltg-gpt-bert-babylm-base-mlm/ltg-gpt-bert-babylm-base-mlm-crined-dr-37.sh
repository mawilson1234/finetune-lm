#!/bin/bash

#SBATCH --job-name=ltg-gpt-bert-babylm-base-mlm-crined-dr-37
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a5000|a100"
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/for_human_exp/ltg-gpt-bert-babylm-base-mlm/ltg-gpt-bert-babylm-base-mlm-crined-dr-37.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path ltg/gpt-bert-babylm-base \
	--model_kwargs.trust_remote_code \
	--config_kwargs.trust_remote_code \
	--tokenizer_kwargs.trust_remote_code \
	--use_gpu \
	--model_callbacks.pre_training model_modifiers.SetEncoderModeCallback \
	--train_file "'data/crined_transitive_SVO-OSV_dr_for_human_exp/crined_transitive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--validation_file "'data/crined_passive_SVO-OSV_dr_for_human_exp/crined_passive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--data_preprocessing_fn.train data_preprocessing.mask_word_tokens \
	--data_preprocessing_fn_kwargs.train.word_tokens_to_mask man boy woman girl water milk wine beer \
	--data_preprocessing_fn.validation data_preprocessing.mask_word_tokens \
	--data_preprocessing_fn_kwargs.validation.word_tokens_to_mask man boy woman girl water milk wine beer \
	--patience 30 \
	--epochs 1000 \
	--min_epochs 100 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.model_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.config_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn data_preprocessing.mask_random_tokens \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn_strategy per_batch \
	--loss_classes_kwargs.train.KLBaselineLoss.tokenizer_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.scaleby 257 \
	--train_optimizer_kwargs.lr 3e-04 \
	--model_callbacks.pre_test model_modifiers.SetEncoderModeCallback \
	--data_preprocessing_fn.test data_preprocessing.mask_word_tokens \
	--data_preprocessing_fn_kwargs.test.word_tokens_to_mask \
		man boy woman girl water milk wine beer \
		teenager dog cat gentleman friend worker jar glass dish mirror window camera \
		customer guest manager farmer doctor artist snack tape phone grain medicine hammer \
		man boy woman girl person customer water milk beer coffee tea \
		person guest child customer patient king pizza orange snack bread sandwich cake \
		journalist editor student scientist child actor book article document script story magazine \
		prince president client author committee representative choice error decision statement purchase incident \
	--test_file "'data/syn_crined_ext_dr_for_human_exp/syn_crined_ext_dr_for_human_exp.txt.gz'" \
				"'data/combined_fillers_for_human_exp_gpt-bert/combined_fillers_for_human_exp_gpt-bert.txt.gz'" \
	--seed 37 \
	--save_best_model_state_to_disk False \
	--output_dir outputs/crined_transitive_SVO-OSV_dr_for_human_exp/ltg-gpt-bert-babylm-base-mlm/{now}