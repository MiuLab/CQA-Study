set -e
cd src/


# Table 1
for seed in {524,525,526};
do
    for model in {bert,no-text,no-conv}; do
        model_dir="../bert-models/${seed}/${model}"
        cmd=(python test.py
             "${model_dir}/config.json"
             "${model_dir}/model.pkl.2"
             "../data/quac-bert/valid.pkl"
             "${model_dir}/predict-2.json"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-log-2.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-log-2.txt"
    done
done


for seed in {524,525,526};
do
    for model in {bert,no-position}; do
        model_dir="../bert-models/${seed}/${model}"

        # Table 4: with attack
        cmd=(python test.py
             "${model_dir}/config.json"
             "${model_dir}/model.pkl.2"
             "../data/quac-bert-attack/valid.pkl"
             "${model_dir}/predict-attack-2.json"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-attack-log-2.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-attack-log-2.txt"

        # Table 5: with mask
        cmd=(python test.py
             "${model_dir}/config-mask.json"
             "${model_dir}/model.pkl.2"
             "../data/quac-bert/valid.pkl"
             "${model_dir}/predict-mask-2.json"
            )
        echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-mask-log-2.txt"
        ${cmd[@]} | tee -a "${model_dir}/predict-mask-log-2.txt"
    done

    # Table 6: without position information
    model_dir="../bert-models/${seed}/bert"
    cmd=(python test.py
         "${model_dir}/config-rm-ind.json"
         "${model_dir}/model.pkl.2"
         "../data/quac-bert/valid.pkl"
         "${model_dir}/predict-rm-2.json"
        )
    echo "Running command ${cmd[@]}" | tee "${model_dir}/predict-rm-log-2.txt"
    ${cmd[@]} | tee -a "${model_dir}/predict-rm-log-2.txt"
done

