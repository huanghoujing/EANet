# This script demonstrates a specific experiment in the paper.
# It should be run under ${project_dir}.
# Example #1, train GlobalPool on market1501:
#       exp_root=exp/eanet gpu=0 task=GlobalPool source_name=market1501 bash script/exp/train.sh
# Example #2, train PAP_ST_PS for market1501->duke:
#       exp_root=exp/eanet gpu=0 task=PAP_ST_PS source_name=market1501 target_name=duke bash script/exp/train.sh
# Example #3, test the GlobalPool model that was trained on market1501. Make sure exp_dir exists and a ckpt.pth is inside it.
#       exp_root=exp/eanet gpu=0 task=GlobalPool source_name=market1501 only_test=True bash script/exp/train.sh

case "${task}" in
    GlobalPool | PCB | PAP_6P | PAP | PAP_S_PS)
        exp_dir=${exp_root}/${task}/${source_name}
        trainer=eanet_trainer
        ow_str="cfg.dataset.train.name = '${source_name}'"
        ;;
    PAP_StC_PS)
        exp_dir=${exp_root}/${task}/${source_name}
        trainer=eanet_trainer
        # Share COCO's train_cuhk03_style for all versions of CUHK03
        if [[ ${source_name} == *"cuhk03"* ]]; then
            style_name=cuhk03
        else
            style_name=${source_name}
        fi
        ow_str="cfg.dataset.train.name = '${source_name}'; cfg.dataset.cd_train.split = 'train_${style_name}_style'"
        ;;
    PAP_ST_PS)
        exp_dir=${exp_root}/${task}/${source_name}_to_${target_name}
        trainer=eanet_trainer
        ow_str="cfg.dataset.train.name = '${source_name}'; cfg.dataset.cd_train.name = '${target_name}'"
        ;;
    PAP_ST_PS_SPGAN)
        exp_dir=${exp_root}/${task}/${source_name}_to_${target_name}
        trainer=eanet_trainer
        ow_str="cfg.dataset.train.name = '${source_name}'; cfg.dataset.train.split = 'train_${target_name}_style'; cfg.dataset.cd_train.name = '${target_name}'"
        ;;
    # NOTE: This task should be run after PAP_ST_PS_SPGAN finishes so that the resulting ckpt.pth is available
    PAP_ST_PS_SPGAN_CFT)
        exp_dir=${exp_root}/${task}/${source_name}_to_${target_name}
        trainer=cft_trainer
        ow_str="cfg.dataset.train.name = '${target_name}'"
        ;;
    *)
        echo "Invalid Task: ${task}!"
        exit
        ;;
esac

# If variable `only_test` exists and its value is `True`, we set this value to the config file
if [ -n "${only_test}" ] && [ "${only_test}" == True ]; then
    ow_str="${ow_str}; cfg.only_test = True"
else
    rm -rf ${exp_dir}  # Remove results of last run
    if [ ${task} == PAP_ST_PS_SPGAN_CFT ]; then
        mkdir -p ${exp_dir}
        cp ${exp_root}/PAP_ST_PS_SPGAN/${source_name}_to_${target_name}/ckpt.pth ${exp_dir}/
    fi
fi

echo ow_str is: "${ow_str}"

CUDA_VISIBLE_DEVICES=${gpu} \
python -m package.optim.${trainer} \
--cfg_file package/config/default.py \
--ow_file paper_configs/${task}.txt \
--ow_str "${ow_str}" \
--exp_dir ${exp_dir}