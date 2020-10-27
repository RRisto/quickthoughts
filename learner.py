import logging
import time
import json
from ast import literal_eval

import torch
from pathlib import Path
from tqdm import tqdm
from pprint import pformat
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import load_pretrained_embeddings
from src.qt_model import QuickThoughts
from src.utils import checkpoint_training, restore_training, safe_pack_sequence, VisdomLinePlotter
from src.data.corpus import Corpus, create_train_eval_corpus
from src.eval import test_performance

_LOGGER = logging.getLogger(__name__)


class QTLearner:
    def __init__(self, checkpoint_dir, embedding, data_path, batch_size, hidden_size, lr, resume,
                 num_epochs, norm_threshold, config_file_name='config.json', optimiser_func=optim.Adam):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_file_name = config_file_name
        self.embedding = embedding
        self.data_path = data_path
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.resume = resume
        self.num_epochs = num_epochs
        self.norm_threshold = norm_threshold
        self.optimizer_func = optimiser_func
        self.WV_MODEL = load_pretrained_embeddings(self.embedding)
        self.vocab = self.WV_MODEL.vocab
        # model, optimizer, and loss function
        self.qt = QuickThoughts(self.WV_MODEL, self.hidden_size)  # .cuda()
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=self.lr)
        self.optimizer = self.optimizer_func(filter(lambda p: p.requires_grad, self.qt.parameters()), lr=self.lr)

    def gather_conf_info(self):
        return {'checkpoint_dir': str(self.checkpoint_dir),
                'embedding': self.embedding,
                'data_path': self.data_path,
                'batch_size': self.batch_size,
                'hidden_size': self.hidden_size,
                'lr': self.lr,
                'resume': self.resume,
                'num_epochs': self.num_epochs,
                'norm_threshold': self.norm_threshold
                }

    def _init_logging(self):
        # setting up training
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.config_filepath = self.checkpoint_dir / self.config_file_name
        self.config_filepath.write_text(str(self.gather_conf_info()))
        _LOGGER.info(pformat(self.gather_conf_info()))
        _LOGGER.info(f"Wrote config to file: {self.config_filepath}")

    def forward_pass(self, qt, data, mode='train'):
        # forward pass
        if mode == 'train':
            enc_f, enc_g = qt(data)
        else:  # in eval mode returns concatenated enc_f and enc_g
            enc = qt(data)
            enc_f, enc_g = enc[:, :self.hidden_size], enc[:, self.hidden_size:]

        # calculate scores
        scores = torch.matmul(enc_f, enc_g.t())

        # zero out when it's the same sentence
        if torch.cuda.is_available():
            mask = torch.eye(len(scores)).cuda().bool()
        else:
            mask = torch.eye(len(scores)).bool()
        scores.masked_fill_(mask, 0)

        # return log scores and target
        block_log_scores = F.log_softmax(scores, dim=1)
        # targets also topelitz matrix
        targets = qt.generate_targets(self.batch_size, offsetlist=[1])
        loss = self.kl_loss(block_log_scores, targets)
        return loss

    def fit_batch(self, qt, data, optimizer, mode='train'):
        loss = self.forward_pass(qt, data, mode)
        if mode == 'train':
            loss.backward()
            # grad clipping
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), self.norm_threshold)
            optimizer.step()
        return loss

    def fit_eval_epoch(self, train_data_iter, eval_data_iter, qt, optimizer, failed_or_skipped_batches, plotter):
        loss_train, failed_or_skipped_batches_train = self.fit_epoch(train_data_iter, failed_or_skipped_batches,
                                                                     optimizer, qt, plotter, mode='train')
        failed_or_skipped_batches += failed_or_skipped_batches_train
        loss_eval, _ = self.fit_epoch(eval_data_iter, failed_or_skipped_batches, optimizer, qt, plotter,
                                      mode='eval')

        return loss_train, loss_eval, failed_or_skipped_batches

    def fit_epoch(self, data_iter, failed_or_skipped_batches, optimizer, qt, plotter, mode='train'):
        if mode == 'train':
            qt.train()
        else:
            qt.eval()

        data_iter_tq = tqdm(data_iter)
        loss = 0
        for i, data in enumerate(data_iter_tq):
            # deal with bad sequences
            if not data:
                failed_or_skipped_batches += 1
                continue

            # zero out
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    data = data.cuda()

                loss = self.fit_batch(qt, data, optimizer, mode)

                data_iter_tq.set_description(
                    "loss {} {:.4f} | failed/skipped {:3d}".format(mode, loss, failed_or_skipped_batches))

            except Exception as e:
                _LOGGER.exception(e)
                failed_or_skipped_batches += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return loss, failed_or_skipped_batches

    def eval_downstream(self, qt, vocab, datasets=['MR']):
        qt.eval()
        accs = []
        for dataset in datasets:
            acc = test_performance(qt, vocab, dataset, 'data', seed=int(time.time()))
            accs.append(acc)
        return accs
        #     plotter.plot('acc', dataset, 'Downstream Accuracy', i, acc, xlabel='seconds')
        # qt.train()

    def create_dataloaders(self, eval_p=0.2):
        bookcorpus_train, bookcorpus_eval = create_train_eval_corpus(self.data_path, self.vocab,
                                                                     eval_p)
        train_iter = DataLoader(bookcorpus_train,
                                batch_size=self.batch_size,
                                num_workers=1,
                                drop_last=True,
                                pin_memory=True,  # send to GPU
                                collate_fn=safe_pack_sequence)

        eval_iter = DataLoader(bookcorpus_eval,
                               batch_size=self.batch_size,
                               num_workers=1,
                               drop_last=True,
                               pin_memory=True,  # send to GPU
                               collate_fn=safe_pack_sequence)
        return train_iter, eval_iter

    def fit(self):
        self._init_logging()

        plotter = VisdomLinePlotter()
        # load in word vectors
        # WV_MODEL = load_pretrained_embeddings(self.embedding)

        # create datasets
        train_iter, eval_iter = self.create_dataloaders()

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # start training
        self.qt = self.qt.train()
        failed_or_skipped_batches = 0
        # last_train_idx = restore_training(self.checkpoint_dir, qt, optimizer) if self.resume else -1
        # start = time.time()
        # block_size = 5
        best_eval_loss = None
        for j in range(self.num_epochs):
            loss_train, loss_eval, failed_or_skipped_batches = self.fit_eval_epoch(train_iter, eval_iter, self.qt,
                                                                                   self.optimizer,
                                                                                   failed_or_skipped_batches, plotter)
            downstream_datasets = ['MR']
            downstream_accs = self.eval_downstream(self.qt, self.vocab, datasets=downstream_datasets)
            if best_eval_loss is None or best_eval_loss > loss_eval:
                best_eval_loss = loss_eval
                checkpoint_training(self.checkpoint_dir, j, self.qt, self.optimizer)

            plotter.plot('loss', 'train', 'Loss train', j, loss_train.item(), xlabel='epoch')
            plotter.plot('loss', 'eval', 'Loss eval', j, loss_eval.item(), xlabel='epoch')
            for i, acc in enumerate(downstream_accs):
                plotter.plot('acc', f'accuracy {downstream_datasets[i]}', 'Downstream accuracy', j, downstream_accs[i],
                             xlabel='epoch')

            # self.fit_epoch(train_iter, failed_or_skipped_batches, optimizer, qt, plotter)
            # self.eval(eval_iter, j, qt, optimizer, WV_MODEL.vocab, plotter)

    def predict(self, text):
        pass

    def check_conf(self):
        pass

    @classmethod
    def create(cls, checkpoint_dir, embedding, data_path, batch_size, hidden_size, lr, resume, num_epochs,
               norm_threshold):
        return cls(checkpoint_dir, embedding, data_path, batch_size, hidden_size, lr, resume, num_epochs,
                   norm_threshold)

    @classmethod
    def create_from_checkpoint(cls, checkpoint_dir, checkpoint_file_name='checkpoint_latest.pth',
                               config_file_name='config.json'):
        checkpoint_dir = Path(checkpoint_dir)
        conf = literal_eval((checkpoint_dir / config_file_name).read_text())
        # todo optimizer to be customizable
        learner = cls(checkpoint_dir, conf['embedding'], conf['data_path'], conf['batch_size'], conf['hidden_size'],
                      conf['lr'], conf['resume'], conf['num_epochs'], conf['norm_threshold'])

        checkpoint_states = torch.load(checkpoint_dir / checkpoint_file_name)
        learner.qt.load_state_dict(checkpoint_states['state_dict'])
        learner.optimizer.load_state_dict(checkpoint_states['optimizer'])

        return learner
