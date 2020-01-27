set -e


function train_quac {
    set -e
    mode=${1}
    seed=${2:=1023}

    case ${mode} in
    # Table 1
	"flowqa")
	    flags=""
	    train_dir="QuAC_data/no-prepend/no-attack"
	    valid_dir="QuAC_data/no-prepend/no-attack"
	    ;;
	"no-conv")
	    flags="--explicit_dialog_ctx 0 --no_dialog_flow --no_hierarchical_query"
	    train_dir="QuAC_data/no-prepend/no-attack"
	    valid_dir="QuAC_data/no-prepend/no-attack"
	    ;;
	"no-text")
	    flags="--mask_prev_ans --no_hierarchical_query --no_dialog_flow"
	    train_dir="QuAC_data/no-prepend/no-attack"
	    valid_dir="QuAC_data/no-prepend/no-attack"
	    ;;
    # Table 4, 5
	"no-position")
	    flags="--explicit_dialog_ctx 0"
	    train_dir="QuAC_data/no-prepend/no-attack"
	    valid_dir="QuAC_data/no-prepend/no-attack"
	    ;;
	*)
	    echo "No matched mode!"
	    exit 1
	    ;;
    esac
    model_dir="quac-models/${seed}/${mode}"

    if [[ ! -d "${model_dir}" ]]
    then
        mkdir -p "${model_dir}"
    else
	echo "Output directory ${model_dir} exist. Abort!"
	exit 2
    fi
    
    cmd=(python train_QuAC.py ${flags}
         --seed ${seed}
	     --model_dir "${model_dir}"
         --train_dir "${train_dir}"
         --dev_dir   "${valid_dir}"
	)
    echo "Running command ${cmd[@]}" | tee "${model_dir}/train-log.txt"
    ${cmd[@]} | tee -a "${model_dir}/train-log.txt"
}


train_quac flowqa 1023
train_quac flowqa 1024
train_quac flowqa 1025

train_quac no-conv 1023
train_quac no-conv 1024
train_quac no-conv 1025

train_quac no-text 1023
train_quac no-text 1024
train_quac no-text 1025

train_quac no-position 1023
train_quac no-position 1024
train_quac no-position 1025
