import logging
import time

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
from src.data.corpus import Corpus
from src.eval import test_performance

_LOGGER = logging.getLogger(__name__)


class QTLearner:
    def __init__(self, checkpoint_dir, embedding, data_path, batch_size, hidden_size, lr, resume,
                 num_epochs, norm_threshold, config_file_name='config.json'):
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

    def gather_conf_info(self):
        return {'checkpoint_dir': self.checkpoint_dir,
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
        self.checkpoint_dir.mkdir()
        self.config_filepath = self.checkpoint_dir / self.config_file_name
        self.config_filepath.write_text(str(self.gather_conf_info()))
        _LOGGER.info(pformat(self.gather_conf_info()))
        _LOGGER.info(f"Wrote config to file: {self.config_filepath}")

    def fit_epoch(self, train_iter, failed_or_skipped_batches, optimizer, qt, plotter):
        temp = tqdm(train_iter)

        for i, data in enumerate(temp):
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

                # forward pass
                enc_f, enc_g = qt(data)

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

                loss.backward()
                # grad clipping
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), self.norm_threshold)
                optimizer.step()

                temp.set_description("loss {:.4f} | failed/skipped {:3d}".format(loss, failed_or_skipped_batches))

                if i % 1 == 0:
                    plotter.plot('loss', 'train', 'Run: {} Loss'.format(str(self.checkpoint_dir).split('/')[-1]), i,
                                 loss.item())

            except Exception as e:
                _LOGGER.exception(e)
                failed_or_skipped_batches += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def eval(self, i, qt, optimizer, vocab, plotter):
        checkpoint_training(self.checkpoint_dir, i, qt, optimizer)
        qt.eval()
        for dataset in ['MR']:
            acc = test_performance(qt, vocab, dataset, 'data', seed=int(time.time()))
            plotter.plot('acc', dataset, 'Downstream Accuracy', i, acc, xlabel='seconds')
        qt.train()

    def create_dataloader(self, pretrained_embeddings):
        bookcorpus = Corpus(self.data_path, pretrained_embeddings.vocab)
        train_iter = DataLoader(bookcorpus,
                                batch_size=self.batch_size,
                                num_workers=1,
                                drop_last=True,
                                pin_memory=True,  # send to GPU
                                collate_fn=safe_pack_sequence)
        return train_iter

    def fit(self):
        self._init_logging()

        plotter = VisdomLinePlotter()
        # load in word vectors
        WV_MODEL = load_pretrained_embeddings(self.embedding)

        # create dataset
        train_iter = self.create_dataloader(WV_MODEL)

        # model, optimizer, and loss function
        qt = QuickThoughts(WV_MODEL, self.hidden_size)  # .cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, qt.parameters()), lr=self.lr)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # start training
        qt = qt.train()
        failed_or_skipped_batches = 0
        # last_train_idx = restore_training(self.checkpoint_dir, qt, optimizer) if self.resume else -1
        # start = time.time()
        # block_size = 5

        for j in range(self.num_epochs):
            self.fit_epoch(train_iter, failed_or_skipped_batches, optimizer, qt, plotter)
            self.eval(j, qt, optimizer, WV_MODEL.vocab, plotter)

    def check_conf(self):
        pass

    @classmethod
    def create(self, checkpoint_dir, embedding, data_path, batch_size, hidden_size, lr, resume, num_epochs,
               norm_threshold):
        return cls(checkpoint_dir, embedding, data_path, batch_size, hidden_size, lr, resume, num_epochs,
                   norm_threshold)

    @classmethod
    def create_from_conf(self, conf):
        pass
