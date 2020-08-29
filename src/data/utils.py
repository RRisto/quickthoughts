from tqdm import tqdm
from gensim.utils import tokenize
import numpy as np

PAD_TOKEN = '<pad>'
PAD_TOKEN_IND = 0  # nice to keep it 0 for pytorch
UNK_TOKEN = '<unk>'
UNK_TOKEN_IND = 1


def prepare_sequence(text, vocab, max_len=50, unk_token=UNK_TOKEN, no_zeros=False):
    pruned_sequence = zip(filter(lambda x: x in vocab, tokenize(text)), range(max_len))
    seq = [vocab.get(x, unk_token) for (x, _) in pruned_sequence]
    # seq = [vocab[x] for (x, _) in pruned_sequence]
    if len(seq) == 0 and no_zeros:
        return [1]
    return seq


# this function should process all.txt and removes all lines that are empty assuming the vocab
def preprocess(read_path, write_path, vocab, max_len=50):
    with open(read_path) as read_file:
        file_length = sum(1 for line in read_file)

    with open(read_path) as read_file, open(write_path, "w+") as write_file:
        write_file.writelines(
            tqdm(filter(lambda x: prepare_sequence(x, vocab, max_len=max_len), read_file), total=file_length))


def get_pretrained_embeddings(pretrained_embeddings, vocab, vector_size):
    words_found = 0

    embeddings = {}
    for i, word in enumerate(vocab):
        try:
            embeddings[word] = pretrained_embeddings[word]
            words_found += 1
        except KeyError:
            embeddings[word] = np.random.normal(scale=0.6, size=(vector_size,))
    return embeddings
