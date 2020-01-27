set -e
cd src/


parent=${1:-"../bert-models"}


for seed in {524,525,526}
do
    for mode in {bert,no-conv,no-position,no-text}
    do
	echo "Traing ${parent}/${seed}/${mode}..."
	python train.py "${parent}/${seed}/${mode}"
    done
done
