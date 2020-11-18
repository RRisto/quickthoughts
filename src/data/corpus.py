import collections
import logging
import torch
from torch.utils.data.dataset import Dataset
from src.data.utils import prepare_sequence

_LOGGER = logging.getLogger(__name__)

UNK_TOKEN = '<unk>'


class Corpus(Dataset):

    def __init__(self, texts, stoi, tokenizer_func, max_len=50, no_zeros=False, unk_token=UNK_TOKEN):
        self.unk_token = unk_token
        if stoi is None:
            self.stoi = self.get_stoi(texts, tokenizer_func, self.unk_token)
        else:
            self.stoi = stoi
        self.texts = texts
        self.tokenizer_func = tokenizer_func
        self.max_len = max_len
        self.no_zeros = no_zeros

    def __getitem__(self, i):
        seq = prepare_sequence(self.texts[i], self.stoi, self.tokenizer_func, max_len=self.max_len,
                               no_zeros=self.no_zeros)
        return torch.LongTensor(seq)

    def __len__(self):
        return len(self.texts)

    def get_stoi(self, texts, tokenizer_func, unk_token='<unk>'):
        tokens = list(set([item for sublist in texts for item in tokenizer_func(sublist)]))
        tokens = [unk_token] + tokens
        stoi = collections.defaultdict(int, {v: k for k, v in enumerate(tokens)})
        return stoi


def create_train_eval_corpus(file_path, tokenizer_func, eval_p=0.2):
    with open(file_path) as f:
        examples = list(f)
        _LOGGER.info("Successfully read {} lines from file: {}".format(len(examples), file_path))
    eval_start_i = int(len(examples) * (1 - eval_p))
    train_corpus = Corpus(examples[:eval_start_i], None, tokenizer_func=tokenizer_func)
    stoi = train_corpus.stoi
    eval_corpus = Corpus(examples[eval_start_i:], stoi, tokenizer_func=tokenizer_func)
    return train_corpus, eval_corpus, stoi
