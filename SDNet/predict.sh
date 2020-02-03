set -e

PARENT=${1:-"experiments"}
SEEDS=(1033 10 256)

for seed in ${SEEDS[@]}; do
    # Table 1
    for model in {sdnet,no-text,no-conv}; do
        model_dir="${PARENT}/${seed}/${model}"
        cmd=(python predict.py
             "${model_dir}/conf"
             "${model_dir}/conf~/run_1/best_model.pt"
             "../data/coqa/dev.json"
             "${model_dir}/predict.json"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-log.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-log.txt"
    done

    for model in {sdnet,no-position}; do
        model_dir="${PARENT}/${seed}/${model}"

        # Table 4: with attack
        cmd=(python predict.py
             "${model_dir}/conf"
             "${model_dir}/conf~/run_1/best_model.pt"
             "../data/coqa/dev-attack.json"
             "${model_dir}/predict-attack.json"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-attack-log.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-attack-log.txt"

        # Table 5: with mask
        cmd=(python predict.py
             "${model_dir}/conf"
             "${model_dir}/conf~/run_1/best_model.pt"
             "../data/coqa/dev.json"
             "${model_dir}/predict-mask.json"
             "-m 2"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-mask-log.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-mask-log.txt"
    done

    # Table 6: without position information
    model_dir="${PARENT}/${seed}/sdnet"
    cmd=(python predict.py
         "${model_dir}/conf"
         "${model_dir}/conf~/run_1/best_model.pt"
         "../data/coqa/dev.json"
         "${model_dir}/predict-rm.json"
         "-i 0"
         "-n"
        )
    echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-rm-log.txt"
    ${cmd[@]} | tee -a "${model_dir}/predict-rm-log.txt"
done
