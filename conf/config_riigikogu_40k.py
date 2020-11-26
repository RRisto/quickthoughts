import os
from torch import optim
from gensim.utils import tokenize
import sentencepiece as spm

from src_custom.eval import test_performances

__base_dir = os.getenv('DIR', os.getcwd())

sp = spm.SentencePieceProcessor()
sp.load('tokenizers/models/riigikogu_40k.model')

CONFIG = {
    'base_dir': __base_dir,
    'data_path': 'data/riigikogu_segments_sep_line_onöy_text.txt',
    'checkpoint_dir': 'checkpoints/riigikogu_40k',
    'context_size': 1,
    # 'batch_size': 400,
    'batch_size': 50,
    'test_batch_size': 1000,
    'norm_threshold': 5.0,
    'hidden_size': 1000,
    'num_epochs': 50,
    'lr': 5e-4,
    'vocab_size': 38258,
    # 'tokenizer_func': tokenize, #gensim default tokenizer
    'tokenizer_func': sp.encode_as_pieces,  # sentencepiece tokenizer
    # 'embedding': 'glove-wiki-gigaword-50',  # default gensim embeddings
    'embedding': 'embedding_models/riigikogu_w2v_40k.kv',  # custom sentencepiece embeddings
    # 'embedding': None,
    'emb_dim': 300,  # needed if embedding is False
    'optimiser_class': optim.Adam,
    'downstream_evaluation_func': test_performances,
    'downstream_eval_datasets': ['ET_sent'],
    'eval_p': 0.2,
    'cuda': False
}