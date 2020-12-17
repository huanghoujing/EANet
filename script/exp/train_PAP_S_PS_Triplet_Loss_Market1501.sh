CUDA_VISIBLE_DEVICES=0 \
python -m package.optim.eanet_trainer \
--cfg_file package/config/default.py \
--ow_file paper_configs/PAP_S_PS_Triplet_Loss_Market1501.txt \
--exp_dir exp/PAP_S_PS_Triplet_Loss/market1501  #--ow_str "cfg.only_test = True"
