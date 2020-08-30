import os
from datetime import datetime

__base_dir = os.getenv('DIR', 'C:/Users/risto/quickthoughts')

CONFIG = {
    'base_dir': __base_dir,
    'vec_path': '{}/data/GoogleNews-vectors-negative300.bin'.format(__base_dir),
    'data_path': '{}/data/cleaned.txt'.format(__base_dir),
    'checkpoint_dir': '{}/checkpoints/{:%m-%d-%H-%M-%S}'.format(__base_dir, datetime.now()),
    'resume': False,
    'context_size': 1,
    # 'batch_size': 400,
    'batch_size': 50,
    'test_batch_size': 1000,
    'norm_threshold': 5.0,
    'hidden_size': 1000,
    'padding_value': 0,
    'num_epochs': 50,
    'lr': 5e-4,
    'vocab_size': 10000,
    # 'embedding': 'embedding_models/telia_w2v.bin'
    'embedding': 'embedding_models/sample_w2v.bin'
}