import os
from torch import optim
import sentencepiece as spm

from src.callbacks.cb_eval import EvalSaveMetrics
from src.callbacks.one_cycle import OneCycle
from src.utils import VisdomLinePlotter
from src_custom.eval import test_performances

__base_dir = os.getenv('DIR', 'C:/Users/risto/quickthoughts')

sp = spm.SentencePieceProcessor()
sp.load('tokenizers/models/dev.model')

CONFIG = {
    'base_dir': __base_dir,
    'data_path': 'data/cleaned.txt',
    'checkpoint_dir': 'checkpoints/dev',
    'context_size': 1,
    # 'batch_size': 400,
    'seq_max_len': 50,
    'batch_size': 10,
    'test_batch_size': 1000,
    'norm_threshold': 5.0,
    'hidden_size': 1000,
    'num_epochs': 50,
    'lr': 5e-4,
    'vocab_size': 10000,
    # 'tokenizer_func': tokenize, #gensim default tokenizer
    'tokenizer_func': sp.encode_as_pieces,  # sentencepiece tokenizer
    # 'embedding': 'glove-wiki-gigaword-50',  # default gensim embeddings
    'embedding': 'embedding_models/dev_vectors.kv',  # custom sentencepiece embeddings
    # 'embedding': None,
    'emb_dim': 300,  # needed if embedding is False
    'optimiser_class': optim.Adam,
    'eval_p': 0.2,
    'cbs': [EvalSaveMetrics(downstream_evaluation_func=test_performances,
                            downstream_eval_datasets=['MR'],
                            plotter=VisdomLinePlotter()),
            OneCycle()],
    # 'cbs': [],
    'cuda': False
}
