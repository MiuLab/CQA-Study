set -e


output_dir="QuAC_data/no-prepend/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_QuAC.py
     --threads 8
     --train_file "../data/quac/train_v0.2.json"
     --dev_file "../data/quac/val_v0.2.json"
     --output_dir "${output_dir}")
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"


output_dir="QuAC_data/no-prepend/attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_QuAC.py
     --threads 8
     --train_file "../data/quac/train_v0.2.json"
     --dev_file "../data/quac/val_v0.2-attack.json"
     --output_dir "${output_dir}")
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
