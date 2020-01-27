import argparse
import pdb
import sys
import traceback
import logging
import json


def main(args):
    with open(args.input_path) as f:
        data = json.load(f)

    for sample in data['data']:
        for paragraph in sample['paragraphs']:
            qas = sorted(paragraph['qas'],
                         key=lambda s: -len(s['orig_answer']['text']))
            for qa in qas:
                answer_start = qa['orig_answer']['answer_start']
                text = qa['orig_answer']['text']
                if text == 'CANNOTANSWER':
                    continue
                paragraph['context'] = repeat_attack(
                    paragraph['context'],
                    answer_start,
                    text
                )

                for aqa in paragraph['qas']:
                    if (aqa['orig_answer']['answer_start'] >= answer_start
                            and aqa['id'] != qa['id']):
                        aqa['orig_answer']['answer_start'] += (len(text) + 1)

            # recheck
            for aqa in paragraph['qas']:
                a_text = aqa['orig_answer']['text']
                a_start = aqa['orig_answer']['answer_start']
                a_end = a_start + len(a_text)
                if paragraph['context'][a_start: a_end] != a_text:
                    logging.warn('mismatch!')
                    breakpoint()

    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=' ', ensure_ascii=False)


def repeat_attack(context, start_index, text):
    assert context[start_index:start_index + len(text)] == text
    context = context[:start_index] + text + ' ' + context[start_index:]
    return context


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('input_path', type=str,
                        help='')
    parser.add_argument('output_path', type=str,
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
