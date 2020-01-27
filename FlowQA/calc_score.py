import argparse
import sys
import logging
import ipdb
import json
import pickle
from pathlib import Path
from scorer import leave_one_out, leave_one_out_max


def main(args):
    # load predictions
    if args.predict_path.suffix == '.pckl':
        with open(args.predict_path, 'rb') as f:
            predicts = sum(pickle.load(f)['predictions'], [])
    elif args.predict_path.suffix == '.json':
        with open(args.predict_path) as f:
            predicts = [json.loads(line)['best_span_str'][0] for line in f]
    else:
        raise NotImplementedError('Unsupported data type!')

    # load answers
    with open(args.dev_path) as f:
        dev = json.load(f)
    truths = [
        {'answers': [answer['text'] for answer in qa['answers']],
         'context': paragraph['context'],
         'followup': qa['followup'],
         'id': qa['id']
        }
        for article in dev['data']
        for paragraph in article['paragraphs']
        for qa in paragraph['qas']
    ]
    assert len(truths) == len(predicts)

    predict_f1s = []
    predict_followup_f1s = []
    for i, (predict, truth) in enumerate(zip(predicts, truths)):
        agreement_f1 = leave_one_out(truth['answers'])
        if agreement_f1 > args.threshold_f1:
            predict_f1 = leave_one_out_max(predict, truth['answers'], truth['context'])
            predict_f1s.append(predict_f1)

            prev_truth = truths[i - 1]
            if (truth['id'].split('#')[0] == prev_truth['id'].split('#')[0]
                  and prev_truth['followup'] == 'y'):
                predict_followup_f1s.append(predict_f1)

    # print(
    #     json.dumps({
    #         'average_f1': sum(predict_f1s) / len(predict_f1s),
    #         'average_followup_f1': sum(predict_followup_f1s) / len(predict_followup_f1s),
    #         'average_non_followup_f1': (
    #             (sum(predict_f1s) - sum(predict_followup_f1s))
    #             / (len(predict_f1s) - len(predict_followup_f1s))
    #         )
    #     })
    # )
    print('{0:.12f}\n{1:.12f}\n{2:.12f}'
          .format(
              sum(predict_f1s) / len(predict_f1s) * 100,
              sum(predict_followup_f1s) / len(predict_followup_f1s) * 100,
              (
                  (sum(predict_f1s) - sum(predict_followup_f1s))
                  / (len(predict_f1s) - len(predict_followup_f1s)) * 100
              )
          )
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('dev_path', type=Path, help='')
    parser.add_argument('predict_path', type=Path, help='')
    parser.add_argument('--threshold_f1', type=float, default=0.4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main(args)
