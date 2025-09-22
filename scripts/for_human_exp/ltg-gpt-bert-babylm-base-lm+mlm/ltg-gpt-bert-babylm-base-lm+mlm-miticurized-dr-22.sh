#!/bin/bash

#SBATCH --job-name=ltg-gpt-bert-babylm-base-lm+mlm-miticurized-dr-22
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm+mlm

echo "Running script scripts/for_human_exp/ltg-gpt-bert-babylm-base-lm+mlm/ltg-gpt-bert-babylm-base-lm+mlm-miticurized-dr-22.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path ltg/gpt-bert-babylm-base \
	--model_kwargs.trust_remote_code \
	--config_kwargs.trust_remote_code \
	--tokenizer_kwargs.trust_remote_code \
	--use_gpu \
	--model_callbacks.begin_epoch model_modifiers.SwitchEncoderDecoderModesToOppositeOfWhenLastCalledCallback \
	--model_callbacks.post_train_batch model_modifiers.SwitchEncoderDecoderModesCallback \
	--model_callbacks_kwargs.post_train_batch.SwitchEncoderDecoderModesCallback.switch_strategy batch \
	--model_callbacks.post_train_all model_modifiers.SetEncoderModeCallback model_modifiers.SwitchEncoderDecoderModesToOppositeOfWhenLastCalledCallback \
	--model_callbacks.post_validation_batch model_modifiers.SwitchEncoderDecoderModesCallback \
	--model_callbacks_kwargs.post_validation_batch.SwitchEncoderDecoderModesCallback.switch_strategy batch \
	--train_file "'data/miticurized_transitive_SVO-OSV_dr_for_human_exp/miticurized_transitive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--validation_file "'data/miticurized_passive_SVO-OSV_dr_for_human_exp/miticurized_passive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--data_preprocessing_fn.train data_preprocessing.identity_if_decoder_mask_if_encoder \
	--data_preprocessing_fn_strategy.train per_batch \
	--data_preprocessing_fn_kwargs.train.mask_fn data_preprocessing.mask_word_tokens \
	--data_preprocessing_fn_kwargs.train.mask_fn_kwargs.word_tokens_to_mask man boy woman girl water milk wine beer \
	--data_preprocessing_fn.validation data_preprocessing.identity_if_decoder_mask_if_encoder \
	--data_preprocessing_fn_strategy.validation per_batch \
	--data_preprocessing_fn_kwargs.validation.mask_fn data_preprocessing.mask_word_tokens \
	--data_preprocessing_fn_kwargs.validation.mask_fn_kwargs.word_tokens_to_mask man boy woman girl water milk wine beer \
	--patience 60 \
	--epochs 2000 \
	--min_epochs 200 \
	--gradient_accumulation_steps 2 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.model_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.config_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn data_preprocessing.identity_if_decoder_mask_if_encoder \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn_strategy per_batch \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn_kwargs.mask_fn data_preprocessing.mask_random_tokens \
	--loss_classes_kwargs.train.KLBaselineLoss.tokenizer_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.scaleby 1.65 \
	--train_optimizer_kwargs.lr 2e-05 \
	--model_callbacks.pre_test model_modifiers.SetEncoderModeCallback \
	--model_callbacks.pre_test_dataset model_modifiers.SwitchEncoderDecoderModesToOppositeOfWhenLastCalledCallback \
	--data_preprocessing_fn.test data_preprocessing.identity_if_decoder_mask_if_encoder \
	--data_preprocessing_fn_strategy.test per_batch \
	--data_preprocessing_fn_kwargs.test.mask_fn data_preprocessing.mask_word_tokens \
	--data_preprocessing_fn_kwargs.test.mask_fn_kwargs.word_tokens_to_mask \
		man boy woman girl water milk wine beer \
		teenager dog cat gentleman friend worker jar glass dish mirror window camera \
		customer guest manager farmer doctor artist snack tape phone grain medicine hammer \
		man boy woman girl person customer water milk beer coffee tea \
		person guest child customer patient king pizza orange snack bread sandwich cake \
		journalist editor student scientist child actor book article document script story magazine \
		prince president client author committee representative choice error decision statement purchase incident \
	--test_file "'data/syn_blorked_SVO-OSV_bv_for_human_exp_gpt-bert/syn_blorked_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_blorked_SVO-OSV_bv_for_human_exp_gpt-bert/syn_blorked_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/combined_fillers_for_human_exp_gpt-bert/combined_fillers_for_human_exp_gpt-bert.txt.gz'" \
				"'data/combined_fillers_for_human_exp_gpt-bert/combined_fillers_for_human_exp_gpt-bert.txt.gz'" \
	--seed 22