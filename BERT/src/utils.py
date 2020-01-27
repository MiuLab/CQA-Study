import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(hyp, ref):
    if ref == 'CANNOTANSWER':
        if hyp == 'CANNOTANSWER':
            return 1
        return 0

    hyp = normalize_answer(hyp).split()
    ref = normalize_answer(ref).split()

    common = Counter(hyp) & Counter(ref)
    n_common = sum(common.values())
    if n_common == 0:
        return 0

    precision = n_common / len(hyp)
    recall = n_common / len(ref)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def leave_one_out_max(hyp, refs):
    f1s = [f1_score(hyp, ref) for ref in refs]
    if len(f1s) == 1:
        return f1s[0]

    f1s = sorted(f1s, reverse=True)
    return (f1s[0] * (len(f1s) - 1) + f1s[1]) / len(f1s)


def handle_cannot(refs):
    num_cannot = 0
    num_spans = 0
    for ref in refs:
        if ref == 'CANNOTANSWER':
            num_cannot += 1
        else:
            num_spans += 1
    if num_cannot >= num_spans:
        refs = ['CANNOTANSWER']
    else:
        refs = [x for x in refs if x != 'CANNOTANSWER']
    return refs


def quac_f1(hyp, refs):
    """Calculate F1 score as QuAC evaluation script does.

    Args:
        hyp (str): Model prediction.
        refs (list of str): Reference answers.

    Returns:
        float: F1 score.
    """

    refs = handle_cannot(refs)

    return leave_one_out_max(hyp, refs)
