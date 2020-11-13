import os
import pickle
import shutil

from learner import QTLearner


# todo temp fix
class Emb:
    def __init__(self, vocab, embeddings):
        self.vocab = vocab
        self.vectors = embeddings

    def save(self, filename):
        """save class as self.name.txt"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """try load self.name.txt"""
        with open(filename, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    if os.path.isdir('checkpoints/dev'):
        shutil.rmtree('checkpoints/dev')
    learner = QTLearner.create_from_conf('src/config_dev.py')
    learner.fit(plot=True)
