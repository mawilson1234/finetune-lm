#!/bin/bash

#SBATCH --job-name=ltg-gpt-bert-babylm-base-lm-baseline
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/crined_drink/ltg-gpt-bert-babylm-base-lm-baseline.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path ltg/gpt-bert-babylm-base \
	--model_kwargs.trust_remote_code \
	--config_kwargs.trust_remote_code \
	--tokenizer_kwargs.trust_remote_code \
	--use_gpu \
	--model_callbacks.pre_test model_modifiers.SetDecoderModeCallback \
	--test_file "'data/syn_blorked_ext_dr_for_human_exp/syn_blorked_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_blorked_SVO-OSV_bv_for_human_exp_gpt-bert/syn_blorked_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_blorked_SVO-OSV_wr_for_human_exp_gpt-bert/syn_blorked_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_crined_ext_dr_for_human_exp/syn_crined_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_crined_SVO-OSV_bv_for_human_exp_gpt-bert/syn_crined_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_crined_SVO-OSV_wr_for_human_exp_gpt-bert/syn_crined_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_dafed_ext_dr_for_human_exp/syn_dafed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_dafed_SVO-OSV_bv_for_human_exp_gpt-bert/syn_dafed_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_dafed_SVO-OSV_wr_for_human_exp_gpt-bert/syn_dafed_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_irriticiated_ext_dr_for_human_exp/syn_irriticiated_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_irriticiated_SVO-OSV_bv_for_human_exp_gpt-bert/syn_irriticiated_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_irriticiated_SVO-OSV_wr_for_human_exp_gpt-bert/syn_irriticiated_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_miticurized_ext_dr_for_human_exp/syn_miticurized_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_miticurized_SVO-OSV_bv_for_human_exp_gpt-bert/syn_miticurized_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_miticurized_SVO-OSV_wr_for_human_exp_gpt-bert/syn_miticurized_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_onstigipated_ext_dr_for_human_exp/syn_onstigipated_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_onstigipated_SVO-OSV_bv_for_human_exp_gpt-bert/syn_onstigipated_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_onstigipated_SVO-OSV_wr_for_human_exp_gpt-bert/syn_onstigipated_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_peagified_ext_dr_for_human_exp/syn_peagified_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_peagified_SVO-OSV_bv_for_human_exp_gpt-bert/syn_peagified_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_peagified_SVO-OSV_wr_for_human_exp_gpt-bert/syn_peagified_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_piliciated_ext_dr_for_human_exp/syn_piliciated_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_piliciated_SVO-OSV_bv_for_human_exp_gpt-bert/syn_piliciated_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_piliciated_SVO-OSV_wr_for_human_exp_gpt-bert/syn_piliciated_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_smeshed_ext_dr_for_human_exp/syn_smeshed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_smeshed_SVO-OSV_bv_for_human_exp_gpt-bert/syn_smeshed_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_smeshed_SVO-OSV_wr_for_human_exp_gpt-bert/syn_smeshed_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_squailed_ext_dr_for_human_exp/syn_squailed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_squailed_SVO-OSV_bv_for_human_exp_gpt-bert/syn_squailed_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_squailed_SVO-OSV_wr_for_human_exp_gpt-bert/syn_squailed_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_unseffed_ext_dr_for_human_exp/syn_unseffed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_unseffed_SVO-OSV_bv_for_human_exp_gpt-bert/syn_unseffed_SVO-OSV_bv_for_human_exp_gpt-bert.txt.gz'" \
				"'data/syn_unseffed_SVO-OSV_wr_for_human_exp_gpt-bert/syn_unseffed_SVO-OSV_wr_for_human_exp_gpt-bert.txt.gz'" \
				"'data/combined_fillers_for_human_exp_gpt-bert/combined_fillers_for_human_exp_gpt-bert.txt.gz'" \
	--output_dir outputs/baseline/ltg-gpt-bert-babylm-base-lm