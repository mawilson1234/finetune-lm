#!/bin/bash

#SBATCH --job-name=ltg-gpt-bert-babylm-base-crined-dr-mlm-optim
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a5000|a100"
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/crined_drink/ltg-gpt-bert-babylm-base-crined-dr-mlm-optim.sh"
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
	--loss_classes_kwargs.train.KLBaselineLoss.tokenizer_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.config_kwargs.trust_remote_code \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn data_preprocessing.mask_random_tokens \
	--loss_classes_kwargs.train.KLBaselineLoss.data_preprocessing_fn_strategy per_batch \
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
	--study_kwargs.storage_kwargs.log_storage_kwargs.file_path "'./outputs/crined_transitive_SVO-OSV_dr_for_human_exp/ltg-gpt-bert-babylm-base/optuna_journal_storage_ltg-gpt-bert-babylm-base-mlm.log'" \
	--study_kwargs.storage_kwargs.log_storage_kwargs.lock_obj optuna.storages.journal.JournalFileOpenLock \
	--study_kwargs.storage_kwargs.log_storage_kwargs.lock_obj_kwargs.filepath "'./outputs/crined_transitive_SVO-OSV_dr_for_human_exp/ltg-gpt-bert-babylm-base/optuna_journal_storage_ltg-gpt-bert-babylm-base-mlm.log'" \
	--study_kwargs.load_if_exists
