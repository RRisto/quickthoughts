import os
from torch import optim

from src.eval import test_performances

__base_dir = os.getenv('DIR', 'C:/Users/risto/quickthoughts')

CONFIG = {
    'base_dir': __base_dir,
    'vec_path': '{}/data/GoogleNews-vectors-negative300.bin'.format(__base_dir),
    'data_path': 'data/cleaned.txt',
    'checkpoint_dir': 'checkpoints/dev',
    'resume': False,
    'context_size': 1,
    # 'batch_size': 400,
    'batch_size': 50,
    'test_batch_size': 1000,
    'norm_threshold': 5.0,
    'hidden_size': 1000,
    'num_epochs': 50,
    'lr': 5e-4,
    'vocab_size': 10000,
    'embedding': 'glove-wiki-gigaword-50',
    # 'embedding': None,
    'emb_dim': 300,  # needed if embedding is False
    'optimiser_class': optim.Adam,
    'downstream_evaluation_func': test_performances,
    'downstream_eval_datasets': ['MR']
}
