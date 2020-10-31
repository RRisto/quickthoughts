from learner import QTLearner
from src.config_dev import CONFIG

if __name__ == '__main__':
    learner = QTLearner.create_from_conf('src/config_dev.py')
    learner.fit()