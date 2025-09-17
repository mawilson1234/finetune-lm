#!/bin/bash

#SBATCH --job-name=ltg-gpt-bert-babylm-base-crined-dr-optim
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/crined_drink/ltg-gpt-bert-babylm-base-crined-dr-optim.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path ltg/gpt-bert-babylm-base \
	--model_kwargs.trust_remote_code \
	--config_kwargs.trust_remote_code \
	--tokenizer_kwargs.trust_remote_code \
	--use_gpu \
	--train_file "'data/crined_transitive_SVO-OSV_dr_for_human_exp/crined_transitive_SVO-OSV_dr_for_human_exp.txt.gz'" \
	--validation_file "'data/crined_passive_SVO-OSV_dr_for_human_exp/crined_passive_SVO-OSV_dr_for_human_exp.txt.gz'" \
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
	--gradient_accumulation_steps 2 `# we use 2 here so that we get update based on one LM batch and one MLM batch each time` \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--loss_classes_kwargs.train.KLBaselineLoss.model_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.tokenizer_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.config_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn data_preprocessing.identity_if_decoder_mask_if_encoder \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn_strategy per_batch \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn_kwargs.mask_fn data_preprocessing.mask_random_tokens \
	--model_callbacks.begin_epoch model_modifiers.SwitchEncoderDecoderModesToOppositeOfWhenLastCalledCallback \
	--model_callbacks.post_train_batch model_modifiers.SwitchEncoderDecoderModesCallback \
	--model_callbacks_kwargs.post_train_batch.SwitchEncoderDecoderModesCallback.switch_strategy batch \
	--model_callbacks.post_train_all model_modifiers.SetEncoderModeCallback model_modifiers.SwitchEncoderDecoderModesToOppositeOfWhenLastCalledCallback \
	--model_callbacks.post_validation_batch model_modifiers.SwitchEncoderDecoderModesCallback \
	--model_callbacks_kwargs.post_validation_batch.SwitchEncoderDecoderModesCallback.switch_strategy batch \
	--do_optimize \
	--max_trials 150 \
	--optimize_kwargs.n_trials 150 \
	--study_kwargs.sampler optuna.samplers.TPESampler \
	--study_kwargs.sampler_kwargs __delete_field__ \
	--study_kwargs.pruner_kwargs.wrapped_pruner_kwargs.n_startup_steps 0 \
	--params.lr.values 1e-9 1e-3 \
	--params.lr.suggest_kwargs.log \
	--params.train_KLBaselineLoss_scaleby.values 1 5000 \
	--params.train_KLBaselineLoss_scaleby.type float \
	--params.train_KLBaselineLoss_scaleby.suggest_kwargs.log \
	--study_kwargs.storage optuna.storages.JournalStorage \
	--study_kwargs.storage_kwargs.log_storage optuna.storages.journal.JournalFileBackend \
	--study_kwargs.storage_kwargs.log_storage_kwargs.file_path "'./outputs/crined_transitive_SVO-OSV_dr_for_human_exp/ltg-gpt-bert-babylm-base/optuna_journal_storage_ltg-gpt-bert-babylm-base.log'" \
	--study_kwargs.storage_kwargs.log_storage_kwargs.lock_obj optuna.storages.journal.JournalFileOpenLock \
	--study_kwargs.storage_kwargs.log_storage_kwargs.lock_obj_kwargs.filepath "'./outputs/crined_transitive_SVO-OSV_dr_for_human_exp/ltg-gpt-bert-babylm-base/optuna_journal_storage_ltg-gpt-bert-babylm-base.log'" \
	--study_kwargs.load_if_exists
