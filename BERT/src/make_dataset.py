import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from preprocessor import Preprocessor


def main(args):
    config_path = os.path.join(args.dest_dir, 'config.json')
    logging.info('loading configuration from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)

    if 'bert' in config and config['bert'] != 'xlnet':
        make_dataset_bert(config)
    elif 'bert' in config and config['bert'] == 'xlnet':
        make_dataset_bert(config, ctx_lm='xlnet')
    else:
        make_dataset(config)


def make_dataset_bert(config, ctx_lm='bert'):
    # init preprocessor
    from bert_preprocessor import BertPreprocessor
    preprocessor = BertPreprocessor(ctx_lm)

    # make valid dataset
    logging.info('preprocessing valid data from {}'
                 .format(config['valid_json_path']))
    valid = preprocessor.get_dataset(
        config['valid_json_path'], True,
        context_truncated_len=config['context_truncated_len'],
        utterance_truncated_len=config['utterance_truncated_len']
    )
    with open(config['valid_pkl_path'], 'wb') as f:
        pickle.dump(valid, f)

    # make train dataset
    logging.info('processing training data from {}'
                 .format(config['train_json_path']))
    train_dataset = preprocessor.get_dataset(
        config['train_json_path'], True,
        context_truncated_len=config['context_truncated_len'],
        utterance_truncated_len=config['utterance_truncated_len']
    )
    with open(config['train_pkl_path'], 'wb') as f:
        pickle.dump(train_dataset, f)


def make_dataset(config):
    # loading load data
    logging.info('loading train from {}'
                 .format(config['train_json_path']))
    with open(config['train_json_path']) as f:
        train = json.load(f)

    logging.info('loading valid from {}'
                 .format(config['valid_json_path']))
    with open(config['valid_json_path']) as f:
        valid = json.load(f)

    # init preprocessor
    preprocessor = Preprocessor()

    # tokenized
    logging.info('tokenizing valid data...')
    tokenized_valid = preprocessor.tokenize_data_parallel(
        valid, n_workers=args.n_workers)
    logging.info('tokenizing training data...')
    tokenized_train = preprocessor.tokenize_data_parallel(
        train, n_workers=args.n_workers)

    # make embedding
    logging.info('loading embeddings from {}...'
                 .format(config['embeddings_vec_path']))
    preprocessor.build_embeddings(
        config['embeddings_vec_path'],
        tokenized_train, tokenized_valid,
        oov_as_unk=config['oov_as_unk']
    )
    with open(config['embeddings_pkl_path'], 'wb') as f:
        pickle.dump(preprocessor.embeddings, f)

    # make dataset
    logging.info('preprocessing valid data...')
    valid = preprocessor.get_dataset(
        tokenized_valid,
        context_truncated_len=config['context_truncated_len'],
        utterance_truncated_len=config['utterance_truncated_len'],
        n_workers=args.n_workers
    )
    with open(config['valid_pkl_path'], 'wb') as f:
        pickle.dump(valid, f)

    logging.info('preprocessing training data...')
    train_dataset = preprocessor.get_dataset(
        tokenized_train,
        context_truncated_len=config['context_truncated_len'],
        utterance_truncated_len=config['utterance_truncated_len'],
        n_workers=args.n_workers
    )
    with open(config['train_pkl_path'], 'wb') as f:
        pickle.dump(train_dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_dir', type=str,
                        help='Path to the output directory, which '
                             'must contain the preprocessing '
                             'configuration config.json')
    parser.add_argument('--n_workers', type=int, default=16)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
