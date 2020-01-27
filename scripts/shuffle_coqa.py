import argparse
import pdb
import sys
import traceback
import logging
import json
import spacy
import random


def main(args):
    with open(args.input_path) as f:
        coqa = json.load(f)

    nlp = spacy.load(
        'en',
        disable=['tokenizer', 'tagger', 'ner', 'parser', 'textcat']
    )
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for sample in coqa['data']:
        sents = list(nlp(sample['story']).sents)
        random.shuffle(sents)
        sample['story'] = ' '.join([sent.text for sent in sents])

    with open(args.output_path, 'w') as f:
        json.dump(coqa, f, indent=' ')


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('input_path', type=str,
                        help='')
    parser.add_argument('output_path', type=str)
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
