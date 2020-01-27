import argparse
import pdb
import sys
import traceback
import logging
import torch
import os
import json
import pickle
from tqdm import tqdm


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    if 'valid' in config['model_parameters']:
        logging.info('loading dev data...')
        with open(config['model_parameters']['valid'], 'rb') as f:
            valid = pickle.load(f)
            valid.data = valid.data[:args.n_data]

    if config['arch'] == 'BiDAF':
        from bidaf_predictor import BiDAFPredictor
        PredictorClass = BiDAFPredictor
    elif config['arch'] == 'QANet':
        from qanet_predictor import QANetPredictor
        PredictorClass = QANetPredictor
    elif config['arch'] == 'BERT':
        from bert_predictor import BERTPredictor
        PredictorClass = BERTPredictor

    predictor = PredictorClass(metrics=[],
                               **config['model_parameters'])

    model_path = os.path.join(
        args.model_dir,
        'model.pkl.{}'.format(args.epoch))

    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)

    logging.info('dumping valid set...')
    dataloader = torch.utils.data.DataLoader(
        valid,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid.collate_fn,
        num_workers=1)

    # ****** create hooks here ******
    hooks = {m: DumpHook()
             for m in ['attn1', 'attn2', 'attn3',
                       'attn4', 'attn5', 'attn6']}

    # ****** add hooks to modules ******
    for i in range(6):
        (predictor.model.bert.encoder.layer[i]
         .attention.self.softmax).register_forward_hook(
             hooks['attn{}'.format(i + 1)].forward_hook
        )

    # predictor.model.mcan.register_forward_hook(
    #     hooks['mcan'].forward_hook
    # )

    # inference the model
    with torch.no_grad():
        for batch in tqdm(dataloader):
            predictor._run_iter(batch, False)
            for hook in hooks.values():
                hook.flush_batch()

    outputs = {k: hook.outputs
               for k, hook in hooks.items()}
    output_path = os.path.join(args.model_dir,
                               'dump.pkl.{}'.format(args.epoch))
    logging.info('Write dump to {}...'.format(output_path))
    with open(output_path, 'wb') as f:
        pickle.dump(outputs, f)


class DumpHook:
    def __init__(self):
        self.outputs = []
        self.batch_outputs = []

    def forward_hook(self, module, inputs, outputs):
        if type(outputs) is tuple:
            outputs = [output.cpu() for output in outputs]
        else:
            outputs = outputs.cpu()
        self.batch_outputs.append(outputs)

    def flush_batch(self):
        self.outputs.append(self.batch_outputs)
        self.batch_outputs = []


def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='')
    parser.add_argument('--n_data', type=int, default=100,
                        help='')
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
