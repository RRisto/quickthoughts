import logging
import torch
from torch.utils.data.dataset import Dataset
from src.data.utils import prepare_sequence

_LOGGER = logging.getLogger(__name__)


class Corpus(Dataset):

    def __init__(self, examples, vocab, tokenizer_func, max_len=50, no_zeros=False):
        self.vocab = vocab
        self.examples = examples
        self.tokenizer_func = tokenizer_func
        self.max_len = max_len
        self.no_zeros = no_zeros

    def __getitem__(self, i):
        seq = prepare_sequence(self.examples[i], self.vocab, self.tokenizer_func, max_len=self.max_len,
                               no_zeros=self.no_zeros)
        return torch.LongTensor(seq)

    def __len__(self):
        return len(self.examples)


def create_train_eval_corpus(file_path, vocab, tokenizer_func, eval_p=0.2):
    with open(file_path) as f:
        examples = list(f)
        _LOGGER.info("Successfully read {} lines from file: {}".format(len(examples), file_path))
    eval_start_i = int(len(examples) * (1 - eval_p))
    train_corpus = Corpus(examples[:eval_start_i], vocab, tokenizer_func=tokenizer_func)
    eval_corpus = Corpus(examples[eval_start_i:], vocab, tokenizer_func=tokenizer_func)
    return train_corpus, eval_corpus
