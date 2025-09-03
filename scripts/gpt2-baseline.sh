#!/bin/bash

#SBATCH --job-name=gpt2-baselines
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND

module load miniconda

source activate finetune-lm

echo "Running script scripts/crined_drink/gpt2-baselines.sh"
echo ""

python core/finetune_lm.py \
	--model_name_or_path gpt2 \
	--use_gpu \
	--test_file "'data/syn_blorked_ext_dr_for_human_exp/syn_blorked_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_blorked_SVO-OSV_bv_for_human_exp/syn_blorked_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_blorked_SVO-OSV_wr_for_human_exp/syn_blorked_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_crined_ext_dr_for_human_exp/syn_crined_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_crined_SVO-OSV_bv_for_human_exp/syn_crined_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_crined_SVO-OSV_wr_for_human_exp/syn_crined_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_dafed_ext_dr_for_human_exp/syn_dafed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_dafed_SVO-OSV_bv_for_human_exp/syn_dafed_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_dafed_SVO-OSV_wr_for_human_exp/syn_dafed_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_irriticiated_ext_dr_for_human_exp/syn_irriticiated_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_irriticiated_SVO-OSV_bv_for_human_exp/syn_irriticiated_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_irriticiated_SVO-OSV_wr_for_human_exp/syn_irriticiated_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_miticurized_ext_dr_for_human_exp/syn_miticurized_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_miticurized_SVO-OSV_bv_for_human_exp/syn_miticurized_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_miticurized_SVO-OSV_wr_for_human_exp/syn_miticurized_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_onstigipated_ext_dr_for_human_exp/syn_onstigipated_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_onstigipated_SVO-OSV_bv_for_human_exp/syn_onstigipated_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_onstigipated_SVO-OSV_wr_for_human_exp/syn_onstigipated_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_peagified_ext_dr_for_human_exp/syn_peagified_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_peagified_SVO-OSV_bv_for_human_exp/syn_peagified_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_peagified_SVO-OSV_wr_for_human_exp/syn_peagified_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_piliciated_ext_dr_for_human_exp/syn_piliciated_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_piliciated_SVO-OSV_bv_for_human_exp/syn_piliciated_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_piliciated_SVO-OSV_wr_for_human_exp/syn_piliciated_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_smeshed_ext_dr_for_human_exp/syn_smeshed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_smeshed_SVO-OSV_bv_for_human_exp/syn_smeshed_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_smeshed_SVO-OSV_wr_for_human_exp/syn_smeshed_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_squailed_ext_dr_for_human_exp/syn_squailed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_squailed_SVO-OSV_bv_for_human_exp/syn_squailed_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_squailed_SVO-OSV_wr_for_human_exp/syn_squailed_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/syn_unseffed_ext_dr_for_human_exp/syn_unseffed_ext_dr_for_human_exp.txt.gz'" \
				"'data/syn_unseffed_SVO-OSV_bv_for_human_exp/syn_unseffed_SVO-OSV_bv_for_human_exp.txt.gz'" \
				"'data/syn_unseffed_SVO-OSV_wr_for_human_exp/syn_unseffed_SVO-OSV_wr_for_human_exp.txt.gz'" \
				"'data/combined_fillers_for_human_exp/combined_fillers_for_human_exp.txt.gz'" \
	--output_dir outputs/baseline/gpt2
