import logging
import torch
from torch.utils.data.dataset import Dataset
from src.data.utils import prepare_sequence

_LOGGER = logging.getLogger(__name__)


class Corpus(Dataset):

    def __init__(self, examples, vocab, max_len=50):
        self.vocab = vocab
        self.examples = examples
        # with open(file_path) as f:
        #   self.examples = list(f)

    # _LOGGER.info("Successfully read {} lines from file: {}".format(len(self.examples), file_path))

    def __getitem__(self, i):
        return torch.LongTensor(prepare_sequence(self.examples[i], self.vocab))

    def __len__(self):
        return len(self.examples)


def create_train_eval_corpus(file_path, vocab, eval_p=0.2):
    with open(file_path) as f:
        examples = list(f)
        _LOGGER.info("Successfully read {} lines from file: {}".format(len(examples), file_path))
    eval_start_i=int(len(examples)*(1-eval_p))
    train_corpus = Corpus(examples[:eval_start_i], vocab)
    eval_corpus = Corpus(examples[eval_start_i:], vocab)
    return train_corpus, eval_corpus
