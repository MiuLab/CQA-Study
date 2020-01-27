set -e


mkdir -p CoQA
cp ../data/coqa/train.json CoQA
cp ../data/coqa/dev.json CoQA
cp ../data/coqa/dev-attack.json CoQA
cp ../data/coqa/dev-shuffle.json CoQA


output_dir="CoQA/prepend/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/dev.json"
     --output_dir "${output_dir}")
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev.json" "${output_dir}/dev.json"


output_dir="CoQA/prepend/attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/dev-attack.json"
     --output_dir "${output_dir}")
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev-attack.json" "${output_dir}/dev.json"


output_dir="CoQA/no-prepend/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/dev.json"
     --output_dir "${output_dir}"
     --no_prepend_answer)
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev.json" "${output_dir}/dev.json"


output_dir="CoQA/no-prepend/attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/dev-attack.json"
     --output_dir "${output_dir}"
     --no_prepend_answer)
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev-attack.json" "${output_dir}/dev.json"


output_dir="CoQA/prepend/shuffle"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/coqa-dev-shuffle.json"
     --output_dir "${output_dir}")
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev-shuffle.json" "${output_dir}/dev.json"


output_dir="CoQA/no-position/no-attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/dev.json"
     --output_dir "${output_dir}"
     --no_position
    )
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev.json" "${output_dir}/dev.json"


output_dir="CoQA/no-position/attack"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/dev-attack.json"
     --output_dir "${output_dir}"
     --no_position)
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev-attack.json" "${output_dir}/dev.json"


output_dir="CoQA/no-position/shuffle"
if [[ ! -d "${output_dir}" ]]
then
    mkdir -p "${output_dir}"
fi
cmd=(python preprocess_CoQA.py
     --threads 8
     --dev_file "CoQA/dev-shuffle.json"
     --output_dir "${output_dir}"
     --no_position
    )
echo "Running command ${cmd[@]}" | tee "${output_dir}/preprocess_log.txt"
${cmd[@]} | tee -a "${output_dir}/preprocess_log.txt"
cp "CoQA/dev.json" "${output_dir}/dev.json"
