from learner import QTLearner
from src.config_dev import CONFIG

if __name__ == '__main__':
    learner = QTLearner.create_from_checkpoint('checkpoints/10-27-19-57-32')

    learner.fit()
