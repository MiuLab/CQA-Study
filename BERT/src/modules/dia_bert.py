import torch
from transformers import BertModel, XLNetModel


class DiaBERT(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, max_seq_len=512, min_window_overlap=128,
                 mask='none',
                 dropout_rate=0.1,
                 fp16=False,
                 yes_no_logits=False,
                 ctx_emb='bert'):
        super(DiaBERT, self).__init__()
        assert min_window_overlap % 2 == 0

        self.ctx_emb = ctx_emb
        if ctx_emb == 'bert':            
            pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert = Bert(768, pretrained_bert, mask)
        elif ctx_emb == 'xlnet':
            self.bert = XLNetModel.from_pretrained('xlnet-base-cased')

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
        slices, reverse_maps = slice_with_window(
            self.max_seq_len, self.min_window_overlap,
            context, context_lens, question_lens
        )
        slices_ai, _ = slice_with_window(
            self.max_seq_len, self.min_window_overlap,
            answer_indicator, context_lens, question_lens
        )

        context_questions = []
        segment_ids = []
        attention_masks = []
        qa_indicators = []
        for rev_map, q, qi, cl, ql in zip(reverse_maps,
                                          question, question_indicator,
                                          context_lens, question_lens):
            for _ in rev_map:
                index = len(context_questions)
                context_questions.append(
                    torch.cat([q[:ql], slices[index]], 0)
                )
                qa_indicators.append(
                    torch.cat([qi[:ql], slices_ai[index]], 0)
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
        qa_indicators = torch.stack(qa_indicators, 0)
        segment_ids = torch.tensor(segment_ids).to(context.device)
        attention_masks = torch.tensor(attention_masks).to(context.device)

        if self.ctx_emb == 'bert':
            bert_out = self.bert(
                context_questions,
                token_type_ids=segment_ids,
                attention_mask=attention_masks,
                answer_indicator=qa_indicators
            )[0][-1]
        else:
            bert_out = self.bert(
                context_questions,
                token_type_ids=qa_indicators,
                attention_mask=attention_masks
            )[0]
        catted_bert_out = []

        slice_logits = self.linear_start_end(bert_out)
        logits = []
        max_context_len = max(context_lens)
        i = 0
        for s, (cl, ql, reverse_map) in enumerate(zip(context_lens,
                                                      question_lens,
                                                      reverse_maps)):
            sample_bert_out = []
            sample_logits = []
            for (start, end) in reverse_map:
                start += ql
                end += ql
                sample_logits.append(slice_logits[i, start:end])
                sample_bert_out.append(bert_out[i, start:end])
                i += 1

            if not self.fp16:
                sample_logits.append(torch.zeros(max_context_len - cl, 2)
                                     .to(slice_logits.device))
                sample_bert_out.append(torch.zeros(max_context_len - cl,
                                                   bert_out.shape[-1])
                                       .to(slice_logits.device))
            else:
                sample_logits.append(torch.zeros(max_context_len - cl, 2)
                                     .half().to(slice_logits.device))
                sample_bert_out.append(torch.zeros(max_context_len - cl,
                                                   bert_out.shape[-1])
                                       .half().to(slice_logits.device))

            logits.append(torch.cat(sample_logits))
            # catted_bert_out.append(torch.cat(sample_bert_out, dim=0))

        logits = torch.stack(logits, 0)
        if self.yesno_mlp:
            catted_bert_out = torch.stack(catted_bert_out)
            yesno_logits = self.yesno_mlp(catted_bert_out.max(1)[0])
            return logits[:, :, 0], logits[:, :, 1], yesno_logits
        else:
            return logits[:, :, 0], logits[:, :, 1], None


class Bert(torch.nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    """
    def __init__(self, hidden_size, pretrained_bert, mask='none'):
        super(Bert, self).__init__()
        self.embeddings = pretrained_bert.embeddings
        self.answer_embeddings = torch.nn.Embedding(20, hidden_size)
        self.encoder = pretrained_bert.encoder
        self.pooler = pretrained_bert.pooler
        self.answer_embeddings.weight.data.normal_(
            mean=0.0,
            std=0.02
        )
        self.mask = mask
        assert mask in ['none', 'allow_previous_one', 'allow_previous_all']

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                answer_indicator=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if answer_indicator is None:
            answer_indicator = torch.zeros_like(input_ids)

        batch_size = answer_indicator.shape[0]

        # make mask
        if self.mask == 'none':
            mask_final = attention_mask.unsqueeze(1).unsqueeze(2) \
                                       .to(dtype=next(self.parameters()).dtype)
        else:
            with torch.no_grad():
                # allow question to attend to self
                mask_self = (
                    answer_indicator.view(batch_size, 1, -1)
                    == answer_indicator.view(batch_size, -1, 1)
                ).long()

                # allow question to attend to previous
                if self.mask == 'allow_previous_one':
                    mask_prev = (
                        answer_indicator.view(batch_size, 1, -1)
                        == answer_indicator.view(batch_size, -1, 1) - 1
                    ).long()
                elif self.mask == 'allow_previous_all':
                    mask_prev = (
                        (
                            answer_indicator.view(batch_size, 1, -1)
                            <= answer_indicator.view(batch_size, -1, 1) - 1
                         ) & (
                            answer_indicator.view(batch_size, 1, -1)
                            > 0
                        )
                    ).long()
                # mask_prev *= (1 - token_type_ids).view(batch_size, 1, -1)
                mask_prev *= (1 - token_type_ids).view(batch_size, -1, 1)

                # allow context to attend to itsef
                mask_context = (
                    token_type_ids.view(batch_size, 1, -1)
                    * token_type_ids.view(batch_size, -1, 1)
                )

                last_and_context = (
                    (answer_indicator
                     == (answer_indicator).max(-1, keepdim=True)[0]).long()
                    + token_type_ids
                )
                mask_cross = (
                    last_and_context.view(batch_size, 1, -1)
                    * last_and_context.view(batch_size, -1, 1)
                ).long()

                extended_attention_mask = \
                    attention_mask.unsqueeze(1).unsqueeze(2)

                mask_final = ((mask_context + mask_prev
                               + mask_self + mask_cross)
                              .unsqueeze(1)
                              * extended_attention_mask) <= 0
                mask_final = (
                    # to fp16/32
                    mask_final.to(dtype=next(self.parameters()).dtype)
                    # exp 0
                    * -10000.0
                )

        # Extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        answer_indicator_embedding = self.answer_embeddings(answer_indicator)
        embedding_all = embedding_output + answer_indicator_embedding
        head_mask = [None] * 12
        encoded_layers = self.encoder(
            embedding_all,
            mask_final,
            head_mask=head_mask
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


def slice_with_window(max_seq_len, min_overlap,
                      context,
                      context_lens, question_lens):
    max_seq_len = min(max_seq_len,
                      max([ql + cl
                           for ql, cl in zip(question_lens, context_lens)]
                          )
                      )
    slices = []
    reverse_maps = []
    for c, cl, ql in zip(context, context_lens, question_lens):
        window = max_seq_len - ql
        stride = max(window - min_overlap, 0)
        if cl <= window:
            slices.append(c[:window])
            if c.shape[0] < window:
                slices[-1] = torch.cat(
                    [slices[-1],
                     torch.tensor([0] * (window - c.shape[0]))
                     .to(context.device)]
                )
            reverse_maps.append([(0, cl)])
        elif cl <= 2 * window - min_overlap:
            slices += [c[:window], c[cl - window:cl]]
            reverse_maps.append(
                [(0, cl // 2), (window - (cl + 1) // 2, window)]
            )
        else:
            reverse_maps.append([(0, window - min_overlap // 2)])
            slices.append(c[:window])
            start = stride
            while start + window < cl:
                slices.append(c[start:start + window])
                reverse_maps[-1].append(
                    (min_overlap // 2,
                     window - min_overlap // 2)
                )
                start = start + stride

            remain = cl - (start + min_overlap // 2)
            slices.append(c[cl - window:cl])
            reverse_maps[-1].append(
                (window - remain, window)
            )

    return slices, reverse_maps
