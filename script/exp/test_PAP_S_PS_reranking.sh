export exp_root=${exp_root:=exp/eanet/test_paper_models}
export gpu=${gpu:=0}
export only_test=True
export task=PAP_S_PS
source_name=market1501 bash script/exp/train.sh
source_name=cuhk03_np_detected_jpg bash script/exp/train.sh
source_name=duke bash script/exp/train.sh
