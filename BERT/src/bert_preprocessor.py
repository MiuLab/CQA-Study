import logging
import json
from tqdm import tqdm
from run_squad import read_squad_examples, convert_examples_to_features
from transformers import BertTokenizer
from dataset import QuACDataset


class BertPreprocessor:
    def __init__(self, ctx_emb='bert'):
        run_squad_logger = logging.getLogger('run_squad')
        run_squad_logger.setLevel(logging.WARNING)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
        self.ctx_emb = ctx_emb

    def get_dataset(self, dataset_path, is_training,
                    context_truncated_len=400,
                    utterance_truncated_len=100):
        examples = read_squad_examples(dataset_path, is_training)

        if self.ctx_emb == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.ctx_emb == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

        features = convert_examples_to_features(
            examples, tokenizer,
            max_seq_length=2500,
            doc_stride=2500,
            max_query_length=2500,
            is_training=is_training
        )

        with open(dataset_path) as f:
            raw_examples = json.load(f)

        # since problems are flatten by convert_examples_to_features
        index_feature = 0

        for example in tqdm(raw_examples['data']):
            for paragraph in example['paragraphs']:
                paragraph['context_raw'] = paragraph['context']

                # Since only `qa_feature.token_to_orig_map` (below) maps token
                # to space-splited-word-level indices in the context,
                # `word_offsets` is required to map space-splited-word-level
                # indices to char-level indices.
                word_offsets = [0]
                for word in paragraph['context'].split(' '):
                    word_offsets.append(len(word) + 1 + word_offsets[-1])

                for index_q, qa in enumerate(paragraph['qas']):
                    qa_feature = features[index_feature]
                    index_feature += 1
                    # in `features[index_feature].segment_ids`, question and
                    # context are concatenated. To seperate them, 0/1 stored
                    # in `segment_ids` are used.
                    question_len = qa_feature.segment_ids.index(1)
                    question = qa_feature.input_ids[:question_len]

                    if index_q == 0:  # do only once for a paragraph
                        context_len = \
                            qa_feature.segment_ids[question_len:].index(0)
                        context = (
                            # [question[0]]  # [CLS] token
                            qa_feature
                            .input_ids[question_len:question_len + context_len]
                        )
                        paragraph['context_offset'] = (
                            # [0]
                            [word_offsets[qa_feature.token_to_orig_map[i]]
                             for i in range(question_len,
                                            question_len + context_len - 1)
                             ]
                            + [len(paragraph['context'])]
                        )
                        paragraph['context_tokenized'] = qa_feature.input_ids
                        paragraph['context'] = context

                    qa['question_tokenized'] = tokenizer.tokenize(
                        qa['question'])
                    qa['question'] = question
                    qa['orig_answer_raw'] = qa['orig_answer']['text']
                    qa['orig_answer_text'] = tokenizer.tokenize(
                        qa['orig_answer_raw'])
                    qa['orig_answer_start'] = qa_feature.start_position - question_len
                    qa['orig_answer_end'] = qa_feature.end_position - question_len
                    assert qa['orig_answer_end'] < len(paragraph['context'])

                    # answer indicator for previous questions
                    qa['answer_indicator'] = [0] * context_len
                    for offset in range(1, min(3 + 1, index_q + 1)):
                        index_prev = index_q - offset
                        start, end = (
                            paragraph['qas'][index_prev]['orig_answer_start'],
                            paragraph['qas'][index_prev]['orig_answer_end'] + 1
                        )
                        qa['answer_indicator'][start:end] = (
                            [offset] * (end - start)
                        )

                    if is_training:
                        for answer in qa['answers']:
                            answer['raw'] = answer['text']
                            answer['text'] = tokenizer.tokenize(answer['text'])

        return QuACDataset(
            raw_examples['data'],
            context_truncated_len=context_truncated_len,
            utterance_truncated_len=utterance_truncated_len,
            padding=0
        )
