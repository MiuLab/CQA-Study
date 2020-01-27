import torch
from base_predictor import BasePredictor
from modules import RecurrentTransformer, NLLLoss, RankLoss


class RTPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """

    def __init__(self, embeddings, n_heads=6,
                 dropout_rate=0.2, dim_ff=512,
                 dim_encoder=102, dim_encoder_ff=256,
                 loss='NLLLoss', margin=0, threshold=None,
                 fine_tune_emb=False, has_info=False, n_blocks=1, **kwargs):
        super(RTPredictor, self).__init__(**kwargs)
        self.has_info = has_info
        dim_embed = embeddings.size(1) + (6 if has_info else 0)
        self.model = RecurrentTransformer(
            dim_embed, n_heads, dropout_rate, dim_ff,
            dim_encoder, dim_encoder_ff, has_emb=fine_tune_emb,
            vol_size=embeddings.size(0), n_blocks=n_blocks
        )
        if fine_tune_emb:
            self.embeddings = self.model.embeddings
        else:
            self.embeddings = torch.nn.Embedding(embeddings.size(0),
                                                 embeddings.size(1))

        self.embeddings.weight = torch.nn.Parameter(embeddings)

        # use cuda
        self.model = self.model.to(self.device)
        self.embeddings = self.embeddings.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = {
            'NLLLoss': NLLLoss(),
            'RankLoss': RankLoss(margin, threshold)
        }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embeddings(batch['context'].to(self.device))
            options = self.embeddings(batch['options'].to(self.device))
            if self.has_info:
                prior = batch['prior'].unsqueeze(-1).to(self.device)
                suggested = batch['suggested'].unsqueeze(-1).to(self.device)
                context = torch.cat([context] + [prior, suggested] * 3, -1)

                prior = batch['option_prior'].unsqueeze(-1).to(self.device)
                suggested = batch['option_suggested'].unsqueeze(-1).to(self.device)
                options = torch.cat([options] + [prior, suggested] * 3, -1)

        logits = self.model.forward(
            context.to(self.device),
            batch['utterance_ends'],
            options.to(self.device),
            batch['option_lens'])
        loss = self.loss(logits, batch['labels'].to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embeddings(batch['context'].to(self.device))
        options = self.embeddings(batch['options'].to(self.device))
        if self.has_info:
            prior = batch['prior'].unsqueeze(-1).to(self.device)
            suggested = batch['suggested'].unsqueeze(-1).to(self.device)
            context = torch.cat([context] + [prior, suggested] * 3, -1)

            prior = batch['option_prior'].unsqueeze(-1).to(self.device)
            suggested = batch['option_suggested'].unsqueeze(-1).to(self.device)
            options = torch.cat([options] + [prior, suggested] * 3, -1)

        logits = self.model.forward(
            context.to(self.device),
            batch['utterance_ends'],
            options.to(self.device),
            batch['option_lens'])
        # predicts = logits.max(-1)[1]
        return logits
