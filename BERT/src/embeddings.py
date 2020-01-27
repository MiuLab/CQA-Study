import re
import torch


class Embeddings:
    """
    Args:
        embedding_path (str): Path where embeddings are loaded from.
        words (None or list): If not None, only load embedding of the words in
            the list.
        oov_as_unk (bool): If argument `words` are provided, whether or not
            treat words in `words` but not in embedding file as `<unk>`. If
            true, OOV will be mapped to the index of `<unk>`. Otherwise,
            embeddings of those OOV will be randomly initialize and their
            indices will be after non-OOV.
        lower (bool): Whether or not lower the words.
    """

    def __init__(self, embedding_path,
                 words=None, oov_as_unk=True, lower=True):
        self.word_dict = {}
        self.embeddings = None
        self.lower = lower
        self.extend(embedding_path, words, oov_as_unk)

        if '<unk>' not in self.word_dict:
            self.add('<unk>')

    def to_index(self, word):
        """
        word (str)

        Return:
             index of the word. If the word is not in `words` and not in the
             embedding file, then index of `<unk>` will be returned.
        """
        if self.lower:
            word = word.lower()

        if word not in self.word_dict:
            return self.word_dict['<unk>']
        else:
            return self.word_dict[word]

    def get_dim(self):
        return self.embeddings.size(1)

    def get_vocabulary_size(self):
        return self.embeddings.size(0)

    def add(self, word, vector=None):
        if self.lower:
            word = word.lower()

        if vector is not None:
            vector = vector.view(1, -1)
        else:
            vector = torch.empty(1, self.get_dim())
            torch.nn.init.uniform_(vector)
        self.embeddings = torch.cat([self.embeddings, vector], 0)
        self.word_dict[word] = len(self.word_dict)

    def extend(self, embeddings_path, words, oov_as_unk=True):
        self._load_embeddings(embeddings_path, set(words))

        if words is not None and not oov_as_unk:
            # initialize word vector for OOV
            for word in words:
                if self.lower:
                    word = word.lower()

                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)

            oov_embeddings = torch.nn.init.uniform_(
                torch.empty(len(self.word_dict) - self.embeddings.size(0),
                            self.embeddings.size(1)))

            self.embeddings = torch.cat([self.embeddings, oov_embeddings], 0)

    def _load_embeddings(self, embedding_path, words):
        embedding = []

        with open(embedding_path) as fp:

            row1 = fp.readline()
            # if the first row is not header
            if not re.match('^[0-9]+ [0-9]+$', row1):
                # seek to 0
                fp.seek(0)
            # otherwise ignore the header

            for i, line in enumerate(fp):
                cols = line.rstrip().split(' ')
                word = cols[0]

                # skip word not in words if words are provided
                if words is not None and word not in words:
                    continue
                elif word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                    embedding.append([float(v) for v in cols[1:]])

        embeddings = torch.tensor(embedding)
        if self.embeddings is not None:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        else:
            self.embeddings = embeddings
