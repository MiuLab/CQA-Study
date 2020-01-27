set -e


function train_coqa {
    set -e
    mode=${1}
    seed=${2:=1023}

    case ${mode} in
    # Table 1
	"flowqa")
	    flags=""
	    train_dir="CoQA/prepend/no-attack"
	    valid_dir="CoQA/prepend/no-attack"
	    ;;
	"no-conv")
	    flags="--explicit_dialog_ctx 0 --no_dialog_flow --no_hierarchical_query"
	    train_dir="CoQA/no-prepend/no-attack"
	    valid_dir="CoQA/no-prepend/no-attack"
	    ;;
	"no-text")
	    flags="--mask_prev_ans --no_dialog_flow --no_hierarchical_query"
	    train_dir="CoQA/no-prepend/no-attack"
	    valid_dir="CoQA/no-prepend/no-attack"
	    ;;
    # Table 4, 5
	"no-position-fix")
	    flags="--explicit_dialog_ctx 0"
	    train_dir="CoQA/no-position/no-attack"
	    valid_dir="CoQA/no-position/no-attack"
	    ;;
	*)
	    echo "No matched mode!"
	    exit 1
	    ;;
    esac
    model_dir="coqa-models/${seed}/${mode}"

    if [[ ! -d "${model_dir}" ]]
    then
        mkdir -p "${model_dir}"
    else
	echo "Output directory ${model_dir} exist. Abort!"
	exit 2
    fi
    
    cmd=(python train_CoQA.py ${flags}
         --seed ${seed}
	     --model_dir "${model_dir}"
         --train_dir "${train_dir}"
         --dev_dir   "${valid_dir}"
	)
    echo "Running command ${cmd[@]}" | tee "${model_dir}/train-log.txt"
    ${cmd[@]} | tee -a "${model_dir}/train-log.txt"
}


train_coqa flowqa  1023
train_coqa flowqa  1024
train_coqa flowqa  1025

train_coqa no-conv 1023
train_coqa no-conv 1024
train_coqa no-conv 1025

train_coqa no-text 1023
train_coqa no-text 1024
train_coqa no-text 1025

train_coqa no-position-fix 1023
train_coqa no-position-fix 1024
train_coqa no-position-fix 1025
