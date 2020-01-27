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
        data = json.load(f)

    nlp = spacy.load(
        'en',
        disable=['tokenizer', 'tagger', 'ner', 'parser', 'textcat']
    )
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for sample in data['data']:
        if args.attack == 'repeat':
            answers = sorted(
                sample['answers'],
                key=lambda a: -len(a['span_text'])
            )
            for answer in answers:
                answer_satrt = answer['span_start']
                text = answer['span_text']
                if text == 'unknown':
                    continue
                sample['story'] = repeat_attack(
                    sample['story'],
                    answer_satrt,
                    text,
                    args.times
                )
                for oans in sample['answers']:
                    if oans['turn_id'] != answer['turn_id'] and \
                       oans['span_start'] >= answer_satrt:
                        oans['span_start'] += (len(text) + 1) * args.times
                        oans['span_end'] += (len(text) + 1) * args.times

        elif args.attack == 'random':
            sample['story'] = random_attack(
                nlp, sample['story'], len(sample['answers'])
            )

        for answer in answers:
            if answer['span_text'] != 'unknown' \
               and answer['span_text'] != (
                sample['story'][answer['span_start']:answer['span_end']]
            ):
                logging.warn('Mismatch!')
                breakpoint()

    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=' ', ensure_ascii=False)


def repeat_attack(context, start_index, text, times=1):
    assert context[start_index:start_index + len(text)] == text
    context = (
        context[:start_index]
        + ' '.join([text] * times)
        + ' ' + context[start_index:])
    return context


def random_attack(nlp, context, n):
    sentences = list(nlp(context).sents)
    random_indices = random.choices(list(range(len(sentences))), k=n)
    attacked = []
    for i, sentence in enumerate(sentences):
        attacked.append(sentence.text)
        if i in random_indices:
            attacked.append(sentence.text)

    return ' '.join(attacked)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('input_path', type=str,
                        help='')
    parser.add_argument('output_path', type=str,
                        help='')
    parser.add_argument('--attack', type=str, default='repeat')
    parser.add_argument('--times', type=int, default=1)
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
