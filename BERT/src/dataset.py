import numpy as np
import torch
from torch.utils.data import Dataset


class QuACDataset(Dataset):
    def __init__(self, data, padding=0,
                 context_truncated_len=400,
                 utterance_truncated_len=100):
        self.padding = padding
        self.context_truncated_len = context_truncated_len
        self.turn_truncated_len = utterance_truncated_len

        self.data = []
        for sample in data:
            for paragraph in sample['paragraphs']:
                for i, qa in enumerate(paragraph['qas']):
                    self.data.append({
                        'id': qa['id'],
                        'context_raw': paragraph['context_raw'],
                        'context_tokenized': paragraph['context_tokenized'],
                        'context': paragraph['context'],
                        'context_offset': paragraph['context_offset'],
                        'context_len': len(paragraph['context']),
                        'followup': 'mny'.index(qa['followup']),
                        'yesno': 'xny'.index(qa['yesno']),
                        'questions_tokenized': [
                            qa['question_tokenized']
                            for qa in paragraph['qas'][:i + 1]
                        ],
                        'questions': [
                            qa['question']
                            for qa in paragraph['qas'][:i + 1]
                        ],
                        'question_lens': [
                            len(qa['question'])
                            for qa in paragraph['qas'][:i + 1]
                        ],
                        'answers_raw': [paragraph['qas'][i]['orig_answer_raw']],
                        'answers': [
                            qa['orig_answer_text']
                            for qa in paragraph['qas'][:i + 1]
                        ],
                        'answer_lens': [
                            len(qa['orig_answer_text'])
                            for qa in paragraph['qas'][:i + 1]
                        ],
                        'answer_spans': [
                            [qa['orig_answer_start'], qa['orig_answer_end']]
                            for qa in paragraph['qas'][:i + 1]
                        ],
                        'answer_indicator': qa['answer_indicator'],
                        'turn_offset': i
                    })
                    if 'answers' in qa:
                        self.data[-1]['answers_raw'] += \
                            [ans['raw'] for ans in qa['answers']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, samples):
        samples = [s for s in samples
                   if s['answer_spans'][-1][0] < self.context_truncated_len
                   and s['answer_spans'][-1][1] < self.context_truncated_len]

        batch = {}
        for key in ['id', 'context_raw', 'context_offset', 'context_len',
                    'answers_raw', 'answers', 'answer_lens', 'answer_spans',
                    'question_lens', 'questions', 'turn_offset', 'context_tokenized',
                    'questions_tokenized']:
            if key == 'answers' and (len(samples) == 0 or key not in samples[0]):
                continue

            batch[key] = [sample[key]
                          for sample in samples]

        for key in ['followup', 'yesno']:
            batch[key] = torch.tensor(
                [sample[key]
                 for sample in samples]
            ).long()

        for key in ['context', 'answer_indicator']:
            # compatible with older code
            if key in ['answer_indicator'] and key not in samples[0]:
                continue

            batch[key] = torch.tensor([
                pad_to_len(
                    sample[key],
                    min(
                        self.context_truncated_len,
                        max(batch['context_len'])
                    ),
                    self.padding
                )
                for sample in samples
            ]).long()

        for key in ['context_tokenized']:
            batch[key] = [sample['context_tokenized'][:self.context_truncated_len]
                          for sample in samples]

        return batch


def pad_to_len(seq, padded_len, padding=0):
    """ Pad sequences to min(max(lens), padded_len).
    Sequence will also be truncated if its length is greater than padded_len.

    Args:
        seqs (list)
        padded_len (int)
        padding (int)
    Return:
        seqs (list)
    """
    padded = [padding] * padded_len
    n_copy = min(len(seq), padded_len)
    padded[:n_copy] = seq[:n_copy]
    return padded
