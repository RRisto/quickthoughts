import os
import shutil

from src.learner import QTLearner
from src.utils import Emb

if __name__ == '__main__':
    if os.path.isdir('checkpoints/dev'):
        shutil.rmtree('checkpoints/dev')
    learner = QTLearner.create_from_conf('src/config_dev.py')
    learner.fit(plot=True)
