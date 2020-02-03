set -e

PARENT=${1:-"experiments"}
SEEDS=(1033 10 256)
MODES=(sdnet no-conv no-text no-position)

for seed in ${SEEDS[@]}; do
    for mode in ${MODES[@]}; do
        echo Traing ${PARENT}/seed-${seed}/${mode}...
        python main.py train ${PARENT}/seed-${seed}/${mode}/conf
    done
done
