import argparse
import json
import pickle
import os
import sys
from pathlib import Path

import ipdb
import torch

from Models.SDNetTrainer import SDNetTrainer
from Utils.Arguments import Arguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf_file', type=str, help='Config file')
    parser.add_argument('model_file', type=str, help='Model file')
    parser.add_argument('data_file', type=str, help='Data file')
    parser.add_argument('output_path', type=Path, help='Output json path')
    parser.add_argument(
        '-m','--mask', type=int, default=0, help='Mask prev ans')
    parser.add_argument(
        '-i','--indicator', type=int, default=0, help='Position indicator of prev ans')
    parser.add_argument(
        '-a','--answer-span-in-context', action='store_true',
        help='Use answer span in context feature')
    parser.add_argument(
        '-n','--no-ans-bit', action='store_true',
        help='Do not use prev ans bit feature')
    args = parser.parse_args()

    return vars(args)


def main(conf_file, model_file, data_file, output_path, mask, indicator, answer_span_in_context, no_ans_bit):
    conf_args = Arguments(conf_file)
    opt = conf_args.readArguments()
    opt['cuda'] = torch.cuda.is_available()
    opt['confFile'] = conf_file
    opt['datadir'] = os.path.dirname(conf_file)
    opt['PREV_ANS_MASK'] = mask
    opt['PREV_ANS_INDICATOR'] = indicator

    opt['OFFICIAL'] = True
    opt['OFFICIAL_TEST_FILE'] = data_file
    if answer_span_in_context:
        opt['ANSWER_SPAN_IN_CONTEXT_FEATURE'] = None
    if no_ans_bit:
        opt['NO_PREV_ANS_BIT'] = None
    trainer = SDNetTrainer(opt)
    test_data = trainer.preproc.preprocess('test')
    predictions, confidence, final_json = trainer.official(model_file, test_data)
    with output_path.open(mode='w') as f:
        json.dump(final_json, f)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
