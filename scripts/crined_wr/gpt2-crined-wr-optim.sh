#!/bin/bash

#SBATCH --job-name=gpt2-crined-wr-optim
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/crined_wrink/gpt2-crined-wr-optim.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path gpt2 \
	--use_gpu \
	--train_file "'data/crined_transitive_SVO-OSV_wr_for_human_exp/crined_transitive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--validation_file "'data/crined_passive_SVO-OSV_wr_for_human_exp/crined_passive_SVO-OSV_wr_for_human_exp.txt.gz'" \
	--patience 30 \
	--epochs 1000 \
	--min_epochs 100 \
	--loss_classes.train loss_classes.OutputsDefaultLoss \
						 loss_classes.KLBaselineLoss \
	--loss_classes_kwargs.train.KLBaselineLoss.dataset "'data/miniboki_train/miniboki_train.txt.gz'" \
	--do_optimize \
	--max_trials 150 \
	--optimize_kwargs.n_trials 50 \
	--study_kwargs.sampler optuna.samplers.TPESampler \
	--study_kwargs.sampler_kwargs __delete_field__ \
	--study_kwargs.pruner_kwargs.wrapped_pruner_kwargs.n_startup_steps 0 \
	--params.lr.values 1e-9 1e-3 \
	--params.lr.suggest_kwargs.log \
	--params.train_KLBaselineLoss_scaleby.values 1 5000 \
	--params.train_KLBaselineLoss_scaleby.type int \
	--params.train_KLBaselineloss_scaleby.suggest_kwargs.log \
	--study_kwargs.storage optuna.storages.JournalStorage \
	--study_kwargs.storage_kwargs.log_storage optuna.storages.journal.JournalFileBackend \
	--study_kwargs.storage_kwargs.log_storage_kwargs.file_path "'./outputs/crined_transitive_SVO-OSV_wr_for_human_exp/optuna_journal_storage.log'" \
	--study_kwargs.storage_kwargs.log_storage_kwargs.lock_obj optuna.storages.journal.JournalFileOpenLock \
	--study_kwargs.storage_kwargs.log_storage_kwargs.lock_obj_kwargs.filepath "'./outputs/crined_transitive_SVO-OSV_wr_for_human_exp/optuna_journal_storage.log'" \
	--study_kwargs.load_if_exists
