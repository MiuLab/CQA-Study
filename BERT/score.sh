set -e
cd src/


# Table 1
for seed in {524,525,526};
do
    for model in {bert,no-text,no-conv}; do
        model_dir="../bert-models/${seed}/${model}"
        cmd=(python scorer.py
             --val_file "../../data/quac/val_v0.2.json"
             --model_output "${model_dir}/predict-2.json"
             --o "${model_dir}/score-2.json"
            )
        echo "Running command ${cmd[@]}"
        ${cmd[@]}
    done
done


for seed in {524,525,526};
do
    for model in {bert,no-position}; do
        model_dir="../bert-models/${seed}/${model}"

        # Table 4: with attack
        cmd=(python scorer.py
             --val_file "../../data/quac/val_v0.2.json"
             --model_output "${model_dir}/predict-attack-2.json"
             --o "${model_dir}/score-attack-2.json"
            )
        echo "Running command ${cmd[@]}"
        ${cmd[@]}
        
        # Table 5: with mask
        cmd=(python scorer.py
             --val_file "../../data/quac/val_v0.2.json"
             --model_output "${model_dir}/predict-mask-2.json"
             --o "${model_dir}/score-mask-2.json"
            )
        echo "Running command ${cmd[@]}"
        ${cmd[@]}
    done

    # Table 6: with position information
    model_dir="../models/${seed}/bert"
    cmd=(python scorer.py
         --val_file "../../data/quac/val_v0.2.json"
         --model_output "${model_dir}/predict-rm-2.json"
         --o "${model_dir}/score-rm-2.json"
        )
    echo "Running command ${cmd[@]}"
    ${cmd[@]}
done

