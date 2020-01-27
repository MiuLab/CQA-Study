import math
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from base_predictor import BasePredictor
from modules import DiaBERT
from dataset import pad_to_len


class BERTPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """

    def __init__(self, embeddings, max_span_len=30,
                 max_seq_len=512, min_window_overlap=128,
                 dropout_rate=0.2, loss='NLLLoss',
                 learning_rate=5e-5, warmup_proportion=0.1,
                 t_total=32906 // 2, fine_tune_emb=False,
                 n_prev=2, n_prev_q=-1, mask='none', remove_ans=False,
                 coqa=False, ctx_emb='bert', no_question=False, **kwargs):
        super(BERTPredictor, self).__init__(**kwargs)
        self.n_prev_q = n_prev_q if n_prev_q >= 0 else n_prev
        self.n_prev_a = n_prev
        self.remove_ans = remove_ans
        self.no_question = no_question

        self.model = DiaBERT(max_seq_len, min_window_overlap,
                             dropout_rate=dropout_rate,
                             fp16=self.fp16,
                             mask=mask, yes_no_logits=coqa,
                             ctx_emb=ctx_emb)

        # load pretrained embedding
        self.padding_indice = 0
        self.max_span_len = max_span_len

        # use cuda
        self.model = self.model.to(self.device)
        if self.fp16:
            self.model = self.model.half()

        # make optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        optimizer_grouped_parameters = [
            {'params': [p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        if not self.fp16:
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=learning_rate,
                                      warmup=warmup_proportion,
                                      t_total=t_total)

        else:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

        self.loss_span = torch.nn.CrossEntropyLoss(reduction='none')
        if coqa:
            self.loss_yesno = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_yesno = None
        self.coqa = coqa

    def _run_iter(self, batch, training):
        batch_size, context_len = batch['context'].shape
        logits_start, logits_end, logits_yesno = \
            self._inference(batch, training)
        answer_spans = torch.tensor(
            [spans[-1] for spans in batch['answer_spans']]
        ).long().to(self.device)
        yesno = batch['yesno'].to(self.device)

        loss_start = self.loss_span(logits_start, answer_spans[:, 0])
        loss_end = self.loss_span(logits_end, answer_spans[:, 1])

        predicts = {}
        context_pad_mask = \
            torch.arange(batch['context'].shape[1]).unsqueeze(0) < \
            torch.LongTensor(batch['context_len']).reshape(-1, 1)
        context_pad_mask = context_pad_mask.to(device=self.device)
        logits = logits_start.unsqueeze(2) + logits_end.unsqueeze(1)
        mask = torch.ones_like(logits[0], dtype=torch.uint8)
        mask = mask.triu().tril(diagonal=self.max_span_len - 1)
        mask = torch.stack([mask] * batch_size, dim=0)
        mask = mask * context_pad_mask.unsqueeze(1) * context_pad_mask.unsqueeze(2)
        logits.masked_fill_(mask == 0, -math.inf)
        predicts_span = logits.reshape(batch_size, -1).max(dim=1, keepdim=True)[1]
        predicts_start = predicts_span / context_len
        predicts_end = predicts_span % context_len
        predicts['span'] = torch.cat([predicts_start, predicts_end], dim=-1)
        if self.coqa:
            predict_yesno = logits_yesno.max(-1)[1]
            for i in range(predicts['span'].shape[0]):
                if predict_yesno[i] == 1:
                    # index of "no" in the context
                    predicts['span'][i, 0] = batch['context_len'][i] - 2
                    predicts['span'][i, 1] = batch['context_len'][i] - 2
                if predict_yesno[i] == 2:
                    # index of "yes" in the context
                    predicts['span'][i, 0] = batch['context_len'][i] - 3
                    predicts['span'][i, 1] = batch['context_len'][i] - 3

        # logits_yesno = logits_yesno[torch.arange(batch_size), predicts_end.squeeze()]
        # logits_followup = logits_followup[torch.arange(batch_size), predicts_end.squeeze()]
        # loss_yesno = torch.nn.functional.cross_entropy(
        #     logits_yesno, batch['yesno'].to(device=logits_yesno.device))
        # loss_followup = torch.nn.functional.cross_entropy(
        #     logits_followup, batch['followup'].to(device=logits_followup.device))

        # predicts['yesno'] = logits_yesno.max(dim=-1)[1]
        # predicts['followup'] = logits_followup.max(dim=-1)[1]

        loss_span = loss_start + loss_end

        if self.loss_yesno:
            loss = (
                loss_span * (yesno == 0).float()  # count only when not yesno
                + self.loss_yesno(logits_yesno, yesno)
            ).mean()
        else:
            loss = loss_span.mean()

        return predicts, loss

    def _predict_batch(self, batch):
        batch_size, context_len = batch['context'].shape
        logits_start, logits_end, logits_yesno = \
            self._inference(batch, False)

        predicts = {}
        context_pad_mask = \
            torch.arange(batch['context'].shape[1]).unsqueeze(0) < \
            torch.LongTensor(batch['context_len']).reshape(-1, 1)
        context_pad_mask = context_pad_mask.to(device=self.device)
        logits = logits_start.unsqueeze(2) + logits_end.unsqueeze(1)
        mask = torch.ones_like(logits[0], dtype=torch.uint8)
        mask = mask.triu().tril(diagonal=self.max_span_len - 1)
        mask = torch.stack([mask] * batch_size, dim=0)
        mask = mask * context_pad_mask.unsqueeze(1) * context_pad_mask.unsqueeze(2)
        logits.masked_fill_(mask == 0, -math.inf)
        predicts_span = logits.reshape(batch_size, -1).max(dim=1, keepdim=True)[1]
        predicts_start = predicts_span / context_len
        predicts_end = predicts_span % context_len
        predicts['span'] = torch.cat([predicts_start, predicts_end], dim=-1)
        if self.coqa:
            predict_yesno = logits_yesno.max(-1)[1]
            for i in range(predicts['span'].shape[0]):
                if predict_yesno[i] == 1:
                    # index of "no" in the context
                    predicts['span'][i, 0] = batch['context_len'][i] - 2  
                    predicts['span'][i, 1] = batch['context_len'][i] - 2
                if predict_yesno[i] == 2:
                    # index of "yes" in the context
                    predicts['span'][i, 0] = batch['context_len'][i] - 3
                    predicts['span'][i, 1] = batch['context_len'][i] - 3

        # predicts['yesno'] = logits_yesno.max(dim=-1)[1]
        # predicts['followup'] = logits_followup.max(dim=-1)[1]
        predicts['logit'] = [logit
                             for logit in logits.reshape(batch_size, -1)]

        return predicts

    def _inference(self, batch, training):
        with torch.no_grad():
            question = [sum(qs[-self.n_prev_q - 1:], [])
                        for qs in batch['questions']]
            if self.no_question:
                question = [[] for qa in batch['questions']]

            question_indicator = [
                sum([[i] * len(qs[-1 - i]) for i in range(min(len(qs), self.n_prev_q + 1))],
                    [])[::-1]
                for qs in batch['questions']
            ]
            question_len = [len(q)
                            for q in question]
            context = batch['context'].to(self.device)
            question = torch.tensor(
                [
                 pad_to_len(q, max(question_len), self.padding_indice)
                 for q in question
                 ]
            ).long().to(self.device)
            question_indicator = torch.tensor(
                [
                 pad_to_len(qi, max(question_len), self.padding_indice)
                 for qi in question_indicator
                 ]
            ).long().to(self.device)

            answer_indicator = batch['answer_indicator'].masked_fill(
                batch['answer_indicator'] > self.n_prev_a, 0
            ).to(self.device)

            if self.remove_ans:
                context = context.masked_fill(answer_indicator > 0, 99)

        logits_start, logits_end, logits_yesno = self.model.forward(
            context,
            batch['context_len'],
            answer_indicator,
            question,
            question_indicator,
            question_len
         )
        return logits_start, logits_end, logits_yesno  #, logits_followup


def bert_batch_to_ids(func, seqs):
    lens = list(map(len, seqs))
    tids = [func(seq) for seq in seqs]
    padded = [tid + [0] * (max(lens) - len(tid))
              for tid in tids]
    return lens, padded
