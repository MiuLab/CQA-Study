from utils import quac_f1


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class SimpleF1(Metrics):
    def __init__(self):
        self.name = 'Simple F1'
        self.reset()

    def reset(self):
        self.f1_sum = 0
        self.n = 0

    def update(self, predicts, batch):
        for (start_, end_), (*_, (start, end)) in \
          zip(predicts['span'], batch['answer_spans']):
            predict = set(range(start_, end_ + 1))
            answer = set(range(start, end + 1))
            precision = len(predict & answer) / (len(predict) + 1e-6)
            recall = len(predict & answer) / (len(answer) + 1e-6)
            f1 = precision * recall / (precision + recall + 1e-6) * 2
            self.f1_sum += f1
            self.n += 1

    def get_score(self):
        return self.f1_sum / self.n


class SimpleEM(Metrics):
    def __init__(self):
        self.name = 'Simple EM'
        self.reset()

    def reset(self):
        self.em_sum = 0
        self.n = 0

    def update(self, predicts, batch):
        for (start_, end_), (*_, (start, end)) in \
          zip(predicts['span'], batch['answer_spans']):
            self.n += 1
            if start_ == start and end_ == end:
                self.em_sum += 1

    def get_score(self):
        return self.em_sum / self.n


class QuACF1(Metrics):
    def __init__(self):
        self.name = 'QuAC F1'
        self.reset()

    def reset(self):
        self.f1_sum = 0
        self.n = 0

    def update(self, predicts, batch):
        for p, context_raw, context_offset, context_len, answers_raw in \
                zip(predicts['span'], batch['context_raw'], batch['context_offset'],
                    batch['context_len'], batch['answers_raw']):
            start = context_offset[p[0]]
            end = len(context_raw) \
                if p[1] == context_len - 1 else context_offset[p[1] + 1]
            hyp = context_raw[start:end]
            self.f1_sum += quac_f1(hyp, answers_raw)
            self.n += 1

    def get_score(self):
        return self.f1_sum / self.n


class Acc(Metrics):
    def __init__(self, key):
        self.name = f'Acc({key})'
        self.key = key
        self.reset()

    def reset(self):
        self.acc_sum = 0
        self.n = 0

    def update(self, predicts, batch):
        for pred, ans in zip(predicts[self.key], batch[self.key]):
            self.acc_sum += (pred.cpu() == ans).sum().item()
            self.n += 1

    def get_score(self):
        return self.acc_sum / self.n
