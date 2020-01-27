import copy
import logging
import spacy
import torch
from tqdm import tqdm
from multiprocessing import Pool
from dataset import QuACDataset
from embeddings import Embeddings


class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """

    def __init__(self, embeddings=None):
        self.nlp = spacy.load('en_core_web_sm',
                              disable=['tagger', 'ner', 'parser', 'textcat'])
        self.tokenized_train = None
        self.tokenized_valid = None
        self.embeddings = embeddings

    def tokenize(self, sentence):
        return self.nlp(sentence)

    def get_dataset(self, raw_data,
                    context_truncated_len=400,
                    utterance_truncated_len=100,
                    n_workers=16, **preprocess_args):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Raw data loaded from json file which has been
                tokenized.
            n_workers (int): Number of CPUs to use.
        """
        processed = self.preprocess_dataset(raw_data)
        return QuACDataset(processed,
                           context_truncated_len=context_truncated_len,
                           utterance_truncated_len=utterance_truncated_len,
                           padding=self.embeddings.to_index('<pad>'))

    def preprocess_dataset(self, dataset):
        """
        Args:
            dataset (list of dict): Raw dataset load from json which is
                tokenized.
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset['data']):
            processed.append(self.preprocess_sample(sample))

        return processed

    def preprocess_sample(self, sample, cat=True):
        """ Preprocess single sample in the dataset.

        Args:
            sample (dict)
        Returns:
            processed sample.
        """
        def find_first_greater(indices, n):
            for i, index in enumerate(indices):
                if index >= n:
                    return i
            return i

        sample = copy.copy(sample)

        for paragraph in sample['paragraphs']:
            # character-level index
            paragraph['context_offset'] = [w.idx
                                           for w in paragraph['context']]
            paragraph['context'] = [self.embeddings.to_index(w.text)
                                    for w in paragraph['context']]
            for qa in paragraph['qas']:
                qa['question_tokenized'] = ['<student>'] + qa['question_tokenized']
                qa['question'] = (
                    [self.embeddings.to_index('<student>')] +
                    [self.embeddings.to_index(w.text)
                     for w in qa['question']]
                )
                qa['orig_answer_raw'] = qa['orig_answer']['raw']
                qa['orig_answer_text'] = (
                    [self.embeddings.to_index('<teacher>')] +
                    [self.embeddings.to_index(w.text)
                     for w in qa['orig_answer']['text']]
                )

                # word-level index
                qa['orig_answer_start'] = \
                    find_first_greater(paragraph['context_offset'],
                                       qa['orig_answer']['answer_start'])
                # minus additional 1 for the '<teacher>' tag
                qa['orig_answer_end'] = (qa['orig_answer_start']
                                         + len(qa['orig_answer_text']) - 2)

                if 'answers' in qa:
                    for answer in qa['answers']:
                        answer['text'] = (
                            [self.embeddings.to_index('<teacher>')] +
                            [self.embeddings.to_index(w.text)
                             for w in answer['text']]
                        )
                        answer['start'] = answer['answer_start']
                        answer['end'] = answer['start'] + len(answer['text'])

        return sample

    def build_embeddings(self, embeddings_vec_path, *raw_datas,
                         oov_as_unk=True, lower=True):
        """ Build Embeddings object that includes vector of words in data.

        Args:
            embeddings_vec_path (str): Path to the pretrained word vector file.
                Ex. FastText.
            raw_datas (list of dict): List of raw data **TOKENIZED** with
                tokenize_data load from json file.
            oov_as_unk (bool): Whether or not treat words not in pretrained
                word vectors set as OOVs. Otherwise, OOVs' embeddings will be
                randomly initialized.
        """
        words = {}
        for raw_data in raw_datas:
            words = self._collect_words(raw_data, words)

        self.embeddings = Embeddings(embeddings_vec_path, words, oov_as_unk,
                                     lower=True)
        self.embeddings.add('<pad>',
                            torch.tensor([0.] * self.embeddings.get_dim()))
        self.embeddings.add('<teacher>')
        self.embeddings.add('<student>')
        self.embeddings.add('CANNOTANSWER')

    def _collect_words(self, data, init_words=None):
        """ Collecte words in the data.

        Args:
            data (dict): tokenized raw data.
            init_words (dict): If provided, words will be accumulated on the
                dict.
        """
        logging.info('Building word list...')
        words = init_words if init_words is not None else {}
        for sample in tqdm(data['data']):
            for paragraph in sample['paragraphs']:
                # collect words in context
                for word in paragraph['context']:
                    if word.text not in words:
                        words[word.text] = 0
                    else:
                        words[word.text] += 1

                # collect words in question
                for qa in paragraph['qas']:
                    for word in qa['question']:
                        if word.text not in words:
                            words[word.text] = 0
                        else:
                            words[word.text] += 1

        return words

    def tokenize_data_parallel(self, raw_data, lower=True, n_workers=16):
        """ Tokenize sentence in the dataset in parallel.

        Args:
            data (dict): Raw data loaded from json file.
            lower (bool): Whether to lower all the characters.
            n_workers (int): Number of CPU to use.
        """
        data = raw_data['data']

        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(data) // n_workers + 1) * i
                batch_end = min((len(data) // n_workers + 1) * (i + 1), len(data))
                batch = data[batch_start:batch_end]
                results[i] = pool.apply_async(self.tokenize_data,
                                              [{'data': batch}, lower])

            pool.close()
            pool.join()

        data = []
        for result in results:
            data += result.get()['data']

        return {'data': data}

    def tokenize_data(self, data, lower=True):
        """ Tokenize sentence in the dataset.

        Args:
            data (dict): Raw data loaded from json file.
        """
        data = copy.deepcopy(data)
        for sample in tqdm(data['data']):
            for paragraph in sample['paragraphs']:
                paragraph['context_raw'] = paragraph['context']
                if lower:
                    paragraph['context'] = paragraph['context'].lower()

                # tokenize context
                paragraph['context'] = self.tokenize(paragraph['context'])
                paragraph['context_tokenized'] = [tok.text for tok in paragraph['context']]

                # tokenize question
                for qa in paragraph['qas']:
                    qa['orig_answer']['raw'] = qa['orig_answer']['text']
                    if lower:
                        qa['question'] = qa['question'].lower()
                        qa['orig_answer']['text'] = \
                            qa['orig_answer']['text'].lower()

                    qa['question'] = self.tokenize(qa['question'])
                    qa['question_tokenized'] = [tok.text for tok in qa['question']]
                    qa['orig_answer']['text'] = \
                        self.tokenize(qa['orig_answer']['text'])

                    # tokenize other answers
                    if 'answers' in qa:
                        for answer in qa['answers']:
                            answer['raw'] = answer['text']
                            if lower:
                                answer['text'] = answer['text'].lower()

                            answer['text'] = self.tokenize(answer['text'])

        return data


class BertTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __call__(self, text):
        class Token:
            def __init__(self, text, offset):
                self.text = text
                self.offset = offset

        tokens = self.tokenizer.basic_tokenizer(text)

        # map tokens to their index in the original text
        curr_offset = 0
        offsets = []
        for token in tokens:
            while token[0] != text[curr_offset]:
                curr_offset += 1
            offsets.append(curr_offset)
            curr_offset += len(token)

        # split tokens into smaller sub-tokens with word piece tokenizer
        subtokens = [
            Token(subtoken, offset)
            for token, offset in zip(tokens, offsets)
            for subtoken in self.tokenizer.wordpiece_tokenizer(token)
        ]

        return subtokens
