import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from metrics import SimpleEM, QuACF1


def main(args):
    with open(args.config_path) as f:
        config = json.load(f)

    # logging.info('loading embedding...')
    # with open(config['model_parameters']['embeddings'], 'rb') as f:
    #     embeddings = pickle.load(f)
    #     config['model_parameters']['embeddings'] = embeddings

    if config['arch'] == 'BiDAF':
        from bidaf_predictor import BiDAFPredictor
        PredictorClass = BiDAFPredictor
    elif config['arch'] == 'QANet':
        from qanet_predictor import QANetPredictor
        PredictorClass = QANetPredictor
    elif config['arch'] == 'BERT':
        from bert_predictor import BERTPredictor
        PredictorClass = BERTPredictor

    if config['arch'] != 'XLNet':
        predictor = PredictorClass(
            metrics=[SimpleEM(), QuACF1()],
            **config['model_parameters']
        )
    else:
        from bert_predictor import BERTPredictor
        predictor = BERTPredictor(
            metrics=[SimpleEM(), QuACF1()],
            ctx_emb='xlnet',
            **config['model_parameters']
        )

    predictor = PredictorClass(**config['model_parameters'])
    predictor.load(args.ckpt_path)

    logging.info('loading dev data...')
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    predictions = predictor.predict_dataset(data, collate_fn=data.collate_fn)
    save_predictions(data, predictions, args.output_path)


def save_predictions(data, predictions, output_path):
    output = []
    for d, p in zip(data, predictions):
        start = d['context_offset'][p[0]]
        end = len(d['context_raw']) \
            if p[1] == d['context_len'] - 1 else d['context_offset'][p[1] + 1]
        best_span_str = d['context_raw'][start:end]
        output.append({
            'qid': [d['id']],
            'yesno': ['y'],
            'followup': ['y'],
            'best_span_str': [best_span_str]
        })

    with open(output_path, 'w') as f:
        for o in output:
            f.write(json.dumps(o) + '\n')


def _parse_args():
    parser = argparse.ArgumentParser(description='Testing script.')
    parser.add_argument('config_path', type=str, help='Path to model config.')
    parser.add_argument('ckpt_path', type=str, help='Path to model checkpoint.')
    parser.add_argument('data_path', type=str, help='Path to testing data.')
    parser.add_argument(
        'output_path', type=str, help='Path to model prediction output.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
