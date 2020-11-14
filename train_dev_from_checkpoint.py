from src.learner import QTLearner
from src.utils import Emb

if __name__ == '__main__':
    learner = QTLearner.create_from_checkpoint('checkpoints/dev')
    learner.fit(plot=True)
