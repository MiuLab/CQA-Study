set -e

parent=${1:-"experiments"}
seeds=(1033 10 256)

for seed in ${seeds[@]}; do
    # table 1
    for model in {sdnet,no-text,no-conv}; do
        model_dir="${parent}/${seed}/${model}"
        cmd=(python evaluate-v1.0.py
             --data-file "../data/coqa/dev.json"
             --pred-file "${model_dir}/predict.json"
             --out-file "${model_dir}/score.json"
            )
        echo "running command ${cmd[@]}"
        ${cmd[@]}
    done

    for model in {sdnet,no-position}; do
        model_dir="${parent}/${seed}/${model}"

        # table 4: with attack
        cmd=(python evaluate-v1.0.py
             --data-file "../data/coqa/dev.json"
             --pred-file "${model_dir}/predict-attack.json"
             --out-file "${model_dir}/score-attack.json"
            )
        echo "running command ${cmd[@]}"
        ${cmd[@]}
        
        # table 5: with mask
        cmd=(python evaluate-v1.0.py
             --data-file "../data/coqa/dev.json"
             --pred-file "${model_dir}/predict-mask.json"
             --out-file "${model_dir}/score-mask.json"
            )
        echo "running command ${cmd[@]}"
        ${cmd[@]}
    done

    # table 6: with position information
    model_dir="${parent}/${seed}/sdnet"
    cmd=(python evaluate-v1.0.py
         --data-file "../data/coqa/dev.json"
         --pred-file "${model_dir}/predict-rm.json"
         --out-file "${model_dir}/score-rm.json"
        )
    echo "running command ${cmd[@]}"
    ${cmd[@]}
done
