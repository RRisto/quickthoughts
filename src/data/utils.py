from tqdm import tqdm


def prepare_sequence(text, vocab, tokenizer_func, max_len=50, no_zeros=False):
    pruned_sequence = zip(filter(lambda x: x in vocab, tokenizer_func(text)), range(max_len))
    #todo custom dict doesnt have .index method
    #seq = [vocab[x].index for (x, _) in pruned_sequence]
    seq = [vocab[x] for (x, _) in pruned_sequence]
    if len(seq) == 0 and no_zeros:
        return [1]
    return seq


# this function should process all.txt and removes all lines that are empty assuming the vocab
def preprocess(read_path, write_path, vocab, tokenizer_func, max_len=50):
    # get the length
    with open(read_path) as read_file:
        file_length = sum(1 for line in read_file)

    with open(read_path) as read_file, open(write_path, "w+") as write_file:
        write_file.writelines(
            tqdm(filter(lambda x: prepare_sequence(x, vocab, tokenizer_func, max_len=max_len), read_file),
                 total=file_length))
