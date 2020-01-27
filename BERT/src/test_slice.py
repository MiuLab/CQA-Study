import argparse
import pdb
import sys
import traceback
import logging
import random
import torch
from modules.dia_bert import slice_with_window


def main(args):
    max_seq_len = 512
    min_overlap = 64
    context_lens = [342, 395]  # list(range(10, 450))
    max_context_len = max(context_lens)
    context = [[i for i in range(cl)]
               + [0] * (max_context_len - cl)
               for cl in context_lens]
    context = torch.tensor(context)
    question_lens = [8, 11]  # [14] * len(context_lens)
    question = [[-i for i in range(ql)]
                + [0] * (max(question_lens) - ql)
                for ql in question_lens]
    question = torch.tensor(question)
    slices, reverse_maps = slice_with_window(
        max_seq_len, min_overlap,
        context,
        context_lens, question_lens
    )

    context_questions = []
    segment_ids = []
    attention_masks = []
    for rev_map, q, cl, ql in zip(reverse_maps, question,
                                  context_lens, question_lens):
        for _ in rev_map:
            index = len(context_questions)
            context_questions.append(
                torch.cat([q[:ql], slices[index]], 0)
            )
            if cl < slices[index].shape[0]:
                segment_ids.append([0] * ql + [1] * cl
                                   + [0] * (slices[index].shape[0] - cl))
                attention_masks.append(
                    [1] * (ql + cl) + [0] * (slices[index].shape[0] - cl))
            else:
                segment_ids.append([0] * ql + [1] * slices[index].shape[0])
                attention_masks.append([1] * (ql + slices[index].shape[0]))

    context_questions = torch.stack(context_questions, 0)

    seqs = []
    i = 0
    for s, (cl, ql, rev_map) in enumerate(zip(context_lens,
                                              question_lens,
                                              reverse_maps)):
        seq = []
        for (start, end) in rev_map:
            start += ql
            end += ql
            seq.append(context_questions[i, start:end])
            i += 1

        seq.append(torch.zeros(max_context_len - cl).long())

        seqs.append(torch.cat(seq))
        assert (seqs[-1] == context[s]).all()

    seqs = [seq.tolist() for seq in seqs]
    context = context.tolist()
    for i, (seq, ctx) in enumerate(zip(seqs, context)):
        assert seq == ctx


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('arg1', type=None,
    #                     help='')
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
