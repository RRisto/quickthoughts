import collections
import logging
import torch
from torch.utils.data.dataset import Dataset
from src.data.utils import prepare_sequence, UNK_TOKEN, UNK_TOKEN_IND, PAD_TOKEN, PAD_TOKEN_IND
from gensim.utils import tokenize

_LOGGER = logging.getLogger(__name__)




class BookCorpus(Dataset):

    def __init__(self, file_path, max_vocab=50000, min_freq=1, pad_token=PAD_TOKEN, pad_token_index=PAD_TOKEN_IND,
                 unk_token=UNK_TOKEN, unk_token_index=UNK_TOKEN_IND):
        self.pad_token = pad_token
        self.pad_token_index = pad_token_index
        self.unk_token=unk_token
        self.unk_token_index=unk_token_index
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        with open(file_path) as f:
            self.examples = list(f)
        _LOGGER.info("Successfully read {} lines from file: {}".format(len(self.examples), file_path))
        self.get_vocab()

    def __getitem__(self, i):
        return torch.LongTensor(prepare_sequence(self.examples[i], self.vocab))

    def __len__(self):
        return len(self.examples)

    def get_vocab(self):
        tokens = [list(tokenize(text, lowercase=True)) for text in self.examples]
        tokens = [item for sublist in tokens for item in sublist]
        freq = collections.Counter(tokens)
        self.itos = [o for o, c in freq.most_common(self.max_vocab) if c >= self.min_freq]
        if self.pad_token in self.itos:
            self.itos.remove(self.pad_token)
        self.itos.insert(self.pad_token_index, self.pad_token)
        if self.unk_token in self.itos:
            self.itos.remove(self.unk_token)
        self.itos.insert(self.unk_token_index, self.unk_token)
        self.vocab = collections.defaultdict(int, {v: k for k, v in enumerate(self.itos)})
