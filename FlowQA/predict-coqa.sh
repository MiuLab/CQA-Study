set -e

function predict_coqa {
    set -e
    seed=$1
    model=$2
    mode=$3

    model_path=./coqa-models/${seed}/${model}/best_model.pt
    output_dir=./coqa-models/${seed}/${model}/${mode}

    if [ "${model}" == "no-conv" ] || [ "${mode}" == "no-text" ]
    then
        prepend="no-prepend"
    elif [ "${model}" == "no-position-fix" ]
    then
        prepend='no-position'
    else
        prepend="prepend"
    fi

    case ${mode} in
        "attack")
            dev_dir=CoQA/${prepend}/attack
            ;;
        "no-attack")
            dev_dir=CoQA/${prepend}/no-attack
            ;;
        "masked")
            extra_flags="--mask_prev_ans"
            dev_dir=CoQA/no-prepend/no-attack
            ;;
        "remove-indicator")
            extra_flags="--remove_indicator"
            dev_dir=CoQA/${prepend}/no-attack
            ;;
        "shuffle")
            dev_dir=CoQA/${prepend}/shuffle
            ;;
    esac

    if [[ ! -d "${output_dir}" ]]
    then
        mkdir "${output_dir}"
    else
        echo "${output_dir} existed. Abort!"
        exit 2
    fi

    cmd=(python predict_CoQA.py
         --model "${model_path}"
         --dev_dir "${dev_dir}"
         -o "${output_dir}"
         ${extra_flags})
    echo "Runnig command \"${cmd[@]}\" " | tee "${output_dir}/predict-log.txt"
    ${cmd[@]} | tee -a "${output_dir}/predict-log.txt"
}


# TABLE 1     seed  model        mode
predict_coqa  1023  flowqa       no-attack
predict_coqa  1025  flowqa       no-attack
predict_coqa  1023  no-text      no-attack
predict_coqa  1023  no-conv      no-attack

# seed: 1023
# TABLE 4     seed  model          mode
predict_coqa  1023  flowqa         attack
predict_coqa  1023  no-position-fix  attack
# TABLE 5     seed  model          mode
predict_coqa  1023  flowqa           masked
predict_coqa  1023  no-position-fix  masked
# TABLE 6     seed  model            mode
predict_coqa  1023  flowqa           remove-indicator
predict_coqa  1023  no-position-fix  remove-indicator
# TABLE 7     seed  model            mode
predict_coqa  1023  no-position-fix  shuffle

# seed: 1024
# TABLE 4     seed  model            mode
predict_coqa  1024  flowqa           attack
predict_coqa  1024  no-position-fix  attack
# TABLE 5     seed  model            mode
predict_coqa  1024  flowqa           masked
predict_coqa  1024  no-position-fix  masked
# TABLE 6     seed  model            mode
predict_coqa  1024  flowqa           remove-indicator
predict_coqa  1024  no-position-fix  remove-indicator
# TABLE 7     seed  model            mode
predict_coqa  1024  no-position-fix  shuffle
                                     
# seed: 1025                         
# TABLE 4     seed  model            mode
predict_coqa  1025  flowqa           attack
predict_coqa  1025  no-position-fix  attack
# TABLE 5     seed  model            mode
predict_coqa  1025  flowqa           masked
predict_coqa  1025  no-position-fix  masked
# TABLE 6     seed  model            mode
predict_coqa  1025  flowqa           remove-indicator
predict_coqa  1025  no-position-fix  remove-indicator
# TABLE 7     seed  model            mode
predict_coqa  1025  no-position-fix  shuffle

