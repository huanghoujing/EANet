# This script demonstrates (almost) all experiments in the paper.
# Don't run this script directly -- it takes several GPU days.
# If you would like to run it, it should be run under ${project_dir}, i.e. `cd ${project_dir}; bash script/exp/train_all.sh`
# You can specify only_test=True to test all models.

# Export, so that script/exp/train.sh can access these variables.
# Use default value, if not set in outside scope.
export exp_root=${exp_root:=exp/eanet}
export gpu=${gpu:=0}
export only_test=${only_test:=False}

for task in GlobalPool PCB PAP_6P PAP PAP_S_PS PAP_StC_PS PAP_ST_PS PAP_ST_PS_SPGAN PAP_ST_PS_SPGAN_CFT
do
    export task  # Export, so that it can be accessed in script/exp/train.sh
    case "${task}" in
        GlobalPool | PCB | PAP_6P | PAP | PAP_S_PS | PAP_StC_PS)
            for source_name in market1501 cuhk03_np_detected_jpg duke
            do
                source_name=${source_name} bash script/exp/train.sh
            done
            ;;
        PAP_ST_PS)
            for source_name in market1501 cuhk03_np_detected_jpg duke
            do
                for target_name in market1501 cuhk03_np_detected_jpg duke
                do
                    if [ ${source_name} != ${target_name} ]
                    then
                        source_name=${source_name} target_name=${target_name} bash script/exp/train.sh
                    fi
                done
            done
            ;;
        PAP_ST_PS_SPGAN)
            for source_name in market1501 duke
            do
                for target_name in market1501 duke
                do
                    if [ ${source_name} != ${target_name} ]
                    then
                        source_name=${source_name} target_name=${target_name} bash script/exp/train.sh
                    fi
                done
            done
            ;;
        # NOTE: This task should be run after PAP_ST_PS_SPGAN finishes so that the resulting ckpt.pth is available
        PAP_ST_PS_SPGAN_CFT)
            for source_name in market1501 duke
            do
                for target_name in market1501 duke
                do
                    if [ ${source_name} != ${target_name} ]
                    then
                        source_name=${source_name} target_name=${target_name} bash script/exp/train.sh
                    fi
                done
            done
            ;;
        *)
            echo "Invalid Task: ${task}!"
            exit
            ;;
    esac
done