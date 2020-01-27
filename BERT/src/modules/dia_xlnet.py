import torch
from transformers import XLNetModel


class DiaXLNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, max_seq_len=512, min_window_overlap=128,
                 mask='none',
                 dropout_rate=0.1,
                 fp16=False,
                 yes_no_logits=False,
                 ctx_emb='bert'):
        super().__init__()
        assert min_window_overlap % 2 == 0

        self.ctx_emb = ctx_emb
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')

        self.linear_start_end = torch.nn.Linear(768, 2, bias=False)
        self.max_seq_len = max_seq_len
        self.min_window_overlap = min_window_overlap
        self.fp16 = fp16
        if yes_no_logits:
            self.yesno_mlp = torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 3)
                )
        else:
            self.yesno_mlp = None

    def forward(self, context, context_lens, answer_indicator,
                question, question_indicator, question_lens):
        batch_size, ctx_len = context.shape
        outputs, mems = self.xlnet(context[:, :512],
                                   token_type_ids=answer_indicator[:, :512])
        encoded = [outputs]
        if ctx_len > 512:
            outputs, _ = self.xlnet(
                context[:, 512:1024],
                mems=mems,
                token_type_ids=answer_indicator[:, 512:1024])
            encoded.append(outputs)

        encoded = torch.cat(encoded, 1)
        logits = self.linear_start_end(encoded)

        # if self.yesno_mlp:
        #     # catted_bert_out = torch.stack(catted_bert_out)
        #     # yesno_logits = self.yesno_mlp(catted_bert_out.max(1)[0])
        #     # return logits[:, :, 0], logits[:, :, 1], yesno_logits
        # else:
        return logits[:, :, 0], logits[:, :, 1], None
