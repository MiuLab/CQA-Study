import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import SimpleF1, SimpleEM, QuACF1, Acc


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # logging.info('loading embedding...')
    # with open(config['model_parameters']['embeddings'], 'rb') as f:
    #     embeddings = pickle.load(f)
    #     config['model_parameters']['embeddings'] = embeddings

    logging.info('loading dev data...')
    with open(config['model_parameters']['valid'], 'rb') as f:
        config['model_parameters']['valid'] = pickle.load(f)

    logging.info('loading train data...')
    with open(config['train'], 'rb') as f:
        train = pickle.load(f)

    if 'train_max_len' in config:
        train.data = list(
            filter(lambda s: s['context_len'] < config['train_max_len'],
                   train.data)
        )

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


    if args.load is not None:
        predictor.load(args.load)

    model_checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, 'model.pkl'),
        'loss', 1, 'all'
    )
    metrics_logger = MetricsLogger(
        os.path.join(args.model_dir, 'log.json')
    )

    logging.info('start training!')
    predictor.fit_dataset(train,
                          train.collate_fn,
                          [model_checkpoint, metrics_logger])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--load', default=None, type=str)
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
