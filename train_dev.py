import os
import shutil

from src.learner import QTLearner

if __name__ == '__main__':
    if os.path.isdir('checkpoints/dev'):
        shutil.rmtree('checkpoints/dev')
    learner = QTLearner.create_from_conf('conf/config_dev.py')
    learner.fit(plot=True)
