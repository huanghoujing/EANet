# Run this script under ${project_dir}

# Use default value, if not set in outside scope.
export exp_root=${exp_root:=exp/eanet/test_paper_models}
export gpu=${gpu:=0}
export only_test=True
bash script/exp/train_all.sh