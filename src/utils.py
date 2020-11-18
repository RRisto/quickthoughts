import logging
import os
import pickle

import torch
from visdom import Visdom
from torch.nn.utils.rnn import pack_sequence
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)


def safe_pack_sequence(x):
    try:
        packed_batch = pack_sequence(x, enforce_sorted=False)
        # targets = torch.zeros(len(x), len(x))
        # for i, t1 in enumerate(x):
        # for j in range(i+1, len(x)):
        # targets[i, j] = len(np.setdiff1d(t1.numpy(),x[j].numpy()))
        # targets += targets.t()

        return packed_batch

    except Exception as e:
        _LOGGER.exception(e)


def log_param_info(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            _LOGGER.debug("name: {} size: {}".format(name, param.data.shape))


def checkpoint_training(checkpoint_dir, idx, model, optim, filename="checkpoint_latest"):
    """checkpoint training, saves optimizer, model, and index"""
    checkpoint_dict = {
        'batch': idx,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
    }
    savepath = "{}/{}.pth".format(checkpoint_dir, filename)
    _LOGGER.info("Saving file at location : {}".format(savepath))
    torch.save(checkpoint_dict, savepath)


def restore_training(checkpoint_dir, model, optimizer, filename="checkpoint_latest"):
    """restore training from a checkpoint dir, returns batch"""
    checkpoint = torch.load("{}/{}.pth".format(checkpoint_dir, filename))
    print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    _LOGGER.info("Resuming training from index: {}".format(checkpoint['batch']))
    return checkpoint['batch']


class VisdomLinePlotter(object):

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y, xlabel='batch'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')


class WV_model:
    def __init__(self, vocab, vectors):
        self.vectors = vectors
        self.vocab = vocab


def match_embeddings_stoi(wv_model, vector_size, stoi):
    matrix_len = len(stoi)
    weights_matrix = np.zeros((matrix_len, vector_size))
    words_found = 0

    mean_vector = np.mean(wv_model.vectors, axis=0)

    for i, word in enumerate(stoi):
        try:
            weights_matrix[i] = wv_model[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = mean_vector

    new_wv_model = WV_model(stoi, weights_matrix)
    return new_wv_model


def load_pretrained_embeddings(file_path, stoi):
    if file_path is None:
        return None

    # function to return pretrained embeddings, embeddings should have:
    # embeddings.vocab - dict {word: id}
    # embeddings.vectors - np array of pretrianed wordvectors, position in array is word id in vocab
    if os.path.isfile(file_path):
        WV_MODEL = KeyedVectors.load(file_path)
    else:
        WV_MODEL = api.load(file_path)

    # kep vectors that are in stoi and initialize ones that are not
    WV_MODEL = match_embeddings_stoi(WV_MODEL, WV_MODEL.vector_size, stoi)

    return WV_MODEL
