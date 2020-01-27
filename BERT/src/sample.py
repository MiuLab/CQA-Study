import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
import math
from scorer import f1_score


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
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
            **config['model_parameters']
        )
    else:
        from bert_predictor import BERTPredictor
        predictor = BERTPredictor(
            ctx_emb='xlnet',
            **config['model_parameters']
        )

    ckpt_path = os.path.join(args.model_dir, 'model.pkl.{}'.format(args.epoch))
    logging.info('Loading model from {}...'.format(ckpt_path))
    predictor.load(ckpt_path)

    logging.info('loading data from {}...'.format(config['train']))
    with open(config['train'], 'rb') as f:
        data = pickle.load(f)
        if args.n_samples > 0:
            data.data = data.data[:args.n_samples]
    if 'train_max_len' in config:
        data.data = list(
            filter(lambda s: (s['context_len'] + s['question_lens'][-1]
                              < config['train_max_len']),
                   data.data)
        )

    logits = predictor.inference_dataset(data, collate_fn=data.collate_fn)
    assert(len(logits) == len(data))

    contrast_spans = {}
    for logit, sample in zip(logits, data):
        ctx_len = int(math.sqrt(logit.shape[0]))
        assert ctx_len ** 2 == logit.shape[0]
        for index in logit.topk(args.topk, -1)[1]:
            span = index.item() // ctx_len, index.item() % ctx_len
            predict = span2text(span, sample)
            f1 = f1_score(predict, sample['answers_raw'][0])
            if (f1 < args.threshold
                    and span[1] - span[0] >= args.min_span_length
                    and predict != 'CANNOTANSWER'):
                contrast_spans[sample['id']] = {'span': span, 'f1': f1}
                break

    output_path = os.path.join(args.model_dir,
                               'contrast-{}.json'.format(args.epoch))
    with open(output_path, 'w') as f:
        json.dump(contrast_spans, f)


def span2text(span, sample):
    start = sample['context_offset'][span[0]]
    end = (
        len(sample['context_raw']) if span[1] == sample['context_len'] - 1
        else sample['context_offset'][span[1] + 1]
    )
    text = sample['context_raw'][start:end]
    return text


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
    parser.add_argument('model_dir', type=str, help='')
    parser.add_argument('epoch', type=int)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--min_span_length', type=int, default=2)
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
