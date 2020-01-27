import math
import torch
from base_predictor import BasePredictor
from modules import QANet
from dataset import pad_to_len


class QANetPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """

    def __init__(self, embeddings, n_blocks, d_model, n_convs, kernel_size,
                 n_heads, max_pos_distance, dropout, max_span_len,
                 fine_tune_emb=False, **kwargs):
        super(QANetPredictor, self).__init__(**kwargs)

        vocab_size, emb_dim = embeddings.embeddings.shape
        self.model = QANet(
            vocab_size, emb_dim, n_blocks, d_model, n_convs, kernel_size,
            n_heads, max_pos_distance, dropout)

        if fine_tune_emb:
            self.embeddings = self.model.embeddings
        else:
            self.embeddings = torch.nn.Embedding(*embeddings.embeddings.shape)

        # load pretrained embedding
        self.embeddings.weight = torch.nn.Parameter(embeddings.embeddings)
        self.padding_indice = embeddings.to_index('<pad>')

        self.max_span_len = max_span_len

        # use cuda
        self.model = self.model.to(self.device)
        self.embeddings = self.embeddings.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = torch.nn.CrossEntropyLoss()

    def _run_iter(self, batch, training):
        batch_size, context_len = batch['context'].shape
        logits_start, logits_end = self._inference(batch, training)
        answer_spans = torch.tensor(
            [spans[-1] for spans in batch['answer_spans']]
        ).long().to(self.device)

        loss_start = self.loss(logits_start, answer_spans[:, 0])
        loss_end = self.loss(logits_end, answer_spans[:, 1])
        loss = loss_start + loss_end

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
        predicts = logits.reshape(batch_size, -1).max(dim=1, keepdim=True)[1]
        predicts = torch.cat([predicts / context_len, predicts % context_len], dim=-1)

        # predicts = torch.cat(
        #     [logits_start.max(-1, keepdim=True)[1],
        #      logits_end.max(-1, keepdim=True)[1]],
        #     -1
        # )
        return predicts, loss

    def _predict_batch(self, batch):
        batch_size, context_len = batch['context'].shape
        logits_start, logits_end = self._inference(batch, False)

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
        predicts = logits.reshape(batch_size, -1).max(dim=1, keepdim=True)[1]
        predicts = torch.cat([predicts / context_len, predicts % context_len], dim=-1)

        # predicts = torch.cat([logits_start.max(-1, keepdim=True),
        #                       logits_end.max(-1, keepdim=True)], -1)
        return predicts

    def _inference(self, batch, training):
        context = self.embeddings(batch['context'].to(self.device))
        question_len = [qls[-1]
                        for qls in batch['question_lens']]
        question = self.embeddings(
            torch.tensor(
                [
                    pad_to_len(qs[-1], max(question_len), self.padding_indice)
                    for qs in batch['questions']
                ]
            ).long().to(self.device)
        )

        context_pad_mask = \
            torch.arange(context.shape[1]).unsqueeze(0) < \
            torch.LongTensor(batch['context_len']).reshape(-1, 1)
        context_pad_mask = context_pad_mask.to(device=self.device)

        question_pad_mask = \
            torch.arange(question.shape[1]).unsqueeze(0) < \
            torch.LongTensor(question_len).reshape(-1, 1)
        question_pad_mask = question_pad_mask.to(device=self.device)

        logits_start, logits_end = self.model.forward(
            context, question, context_pad_mask, question_pad_mask)
        return logits_start, logits_end
