import numpy as np


class Vocab:

    def __init__(self, vocab):
        self.vocab = vocab

    def get_pretrained_embeddings(self, pretrained_embeddings, vector_size):
        matrix_len = len(self.vocab)
        weights_matrix = np.zeros((matrix_len, vector_size))
        words_found = 0

        embeddings={}
        for i, word in enumerate(self.vocab):
            try:
                embeddings[word] = pretrained_embeddings[word]
                words_found += 1
            except KeyError:
                embeddings[word] = np.random.normal(scale=0.6, size=(vector_size,))
        # for i, word in enumerate(self.vocab):
        #     try:
        #         weights_matrix[i] = pretrained_embeddings[word]
        #         words_found += 1
        #     except KeyError:
        #         weights_matrix[i] = np.random.normal(scale=0.6, size=(vector_size,))
        return embeddings
