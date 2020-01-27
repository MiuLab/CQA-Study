

# download original QuAC dataset
mkdir -p data/quac/
if [ ! -e "data/quac/train_v0.2.json" ]
then
    wget "https://s3.amazonaws.com/my89public/quac/train_v0.2.json" -O "data/quac/train_v0.2.json"
fi
if [ ! -e "data/quac/val_v0.2.json" ]
then
    wget "https://s3.amazonaws.com/my89public/quac/val_v0.2.json" -O "data/quac/val_v0.2.json"
fi
# apply the repeat attack on the QuAC
python3 scripts/attack_quac.py "data/quac/val_v0.2.json" "data/quac/val_v0.2-attack.json"


# download original CoQA dataset
mkdir -p data/coqa/
if [ ! -e "data/coqa/train.json" ]
then
    wget "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json" -O "data/coqa/train.json"
fi
if [ ! -e "data/coqa/dev.json" ]
then
    wget "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json" -O "data/coqa/dev.json"
fi
# apply the repeat attack on the CoQA
python3 scripts/attack_coqa.py "data/coqa/dev.json" "data/coqa/dev-attack.json"
python3 scripts/shuffle_coqa.py "data/coqa/dev.json" "data/coqa/dev-shuffle.json"
