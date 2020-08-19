import json
import gensim.downloader as api
import torch
from gensim.utils import tokenize
from torch.nn.utils.rnn import pack_sequence

from src.qt_model import QuickThoughts

checkpoint_dir = 'checkpoints/08-16-16-18-52'
# CONFIG=json.load(Path(checkpoint_dir).read_bytes())

# load conf
with open("{}/config.json".format(checkpoint_dir)) as fp:
    CONFIG = json.load(fp)

# load embeddings, model
WV_MODEL = api.load(CONFIG['embedding'])
qt = QuickThoughts(WV_MODEL, hidden_size=CONFIG['hidden_size'])

trained_params = torch.load("{}/checkpoint_latest.pth".format(checkpoint_dir))
qt.load_state_dict(trained_params['state_dict'])
qt.eval()


# preprocess text
def prepare_sequence(text, vocab, max_len=50, no_zeros=False):
    pruned_sequence = zip(filter(lambda x: x in vocab, tokenize(text)), range(max_len))
    seq = [vocab[x].index for (x, _) in pruned_sequence]
    if len(seq) == 0 and no_zeros:
        return [1]
    return seq


text = 'this is random test text'

sequence = list(map(lambda x: torch.LongTensor(prepare_sequence(x, WV_MODEL.vocab, no_zeros=True)), [text]))
packed_sequence = pack_sequence(sequence, enforce_sorted=False)

vector = qt(packed_sequence).cpu().detach().numpy()
print(vector.shape)
