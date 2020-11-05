import logging
import shutil
import time

import torch
from pathlib import Path
from tqdm import tqdm
from pprint import pformat
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.utils import load_pretrained_embeddings
from src.qt_model import QuickThoughts
from src.utils import checkpoint_training, restore_training, safe_pack_sequence, VisdomLinePlotter
from src.data.corpus import create_train_eval_corpus, Corpus
from src.eval import test_performance, test_performances

_LOGGER = logging.getLogger(__name__)


class QTLearner:
    def __init__(self, checkpoint_dir, embedding, data_path, batch_size, hidden_size, lr, resume, num_epochs,
                 norm_threshold, config_file_name='config.json', optimiser_class=optim.Adam,
                 metrics_filename='metrics.txt'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_file_name = config_file_name
        self.metrics_filename = metrics_filename
        self.embedding = embedding
        self.data_path = data_path
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.resume = resume
        self.num_epochs = num_epochs
        self.norm_threshold = norm_threshold
        self.optimizer_class = optimiser_class
        self.WV_MODEL = load_pretrained_embeddings(self.embedding)
        self.vocab = self.WV_MODEL.vocab
        # model, optimizer, and loss function
        self.qt = QuickThoughts(self.WV_MODEL, self.hidden_size)  # .cuda()
        self.optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.qt.parameters()), lr=self.lr)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.test_downstream_task_func = test_performances

    def gather_conf_info(self):
        return {'checkpoint_dir': str(self.checkpoint_dir),
                'embedding': self.embedding,
                'data_path': self.data_path,
                'batch_size': self.batch_size,
                'hidden_size': self.hidden_size,
                'lr': self.lr,
                'resume': self.resume,
                'num_epochs': self.num_epochs,
                'norm_threshold': self.norm_threshold,
                'optimizer_class': self.optimizer_class
                }

    def _init_logging(self):
        _LOGGER.info(pformat(self.gather_conf_info()))

    def forward_pass(self, qt, data, mode='train', return_only_embedding=False):
        # forward pass
        if mode == 'train':
            enc_f, enc_g = qt(data)
        else:  # in eval mode returns concatenated enc_f and enc_g
            enc = qt(data)
            if return_only_embedding:
                return enc.detach().numpy()
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
        return loss, block_log_scores

    def fit_batch(self, qt, data, optimizer=None, mode='train'):
        loss, block_log_scores = self.forward_pass(qt, data, mode)
        if mode == 'train':
            loss.backward()
            # grad clipping
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, qt.parameters()), self.norm_threshold)
            optimizer.step()
        return loss, block_log_scores

    def fit_eval_epoch(self, train_data_iter, eval_data_iter, qt, optimizer, failed_or_skipped_batches):
        loss_train, failed_or_skipped_batches_train = self.fit_epoch(train_data_iter, failed_or_skipped_batches,
                                                                     optimizer, qt, mode='train')
        failed_or_skipped_batches += failed_or_skipped_batches_train
        loss_eval, _ = self.fit_epoch(eval_data_iter, failed_or_skipped_batches, optimizer, qt, mode='eval')

        return loss_train, loss_eval, failed_or_skipped_batches

    def fit_epoch(self, data_iter, failed_or_skipped_batches, optimizer, qt, mode='train'):
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

                loss, _ = self.fit_batch(qt, data, optimizer, mode)

                data_iter_tq.set_description(
                    "loss {} {:.4f} | failed/skipped {:3d}".format(mode, loss, failed_or_skipped_batches))

            except Exception as e:
                _LOGGER.exception(e)
                failed_or_skipped_batches += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return loss, failed_or_skipped_batches

    def eval_downstream(self, qt, vocab, loc='data', datasets=['MR']):
        # todo make params default to conf
        qt.eval()
        accs = self.test_downstream_task_func(self.predict, datasets=datasets, loc=loc)
        # accs = []
        # for dataset in datasets:
        #   acc = test_performance(qt, vocab, dataset, 'data', seed=int(time.time()))
        #  accs.append(acc)
        return accs

    def create_dataloaders(self, eval_p=0.2):
        bookcorpus_train, bookcorpus_eval = create_train_eval_corpus(self.data_path, self.vocab, eval_p)
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

        # create datasets
        train_iter, eval_iter = self.create_dataloaders()

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
                                                                                   failed_or_skipped_batches)
            downstream_datasets = ['MR']
            downstream_accs = self.eval_downstream(self.qt, self.vocab, datasets=downstream_datasets)
            if best_eval_loss is None or best_eval_loss > loss_eval:
                best_eval_loss = loss_eval
                checkpoint_training(self.checkpoint_dir, j, self.qt, self.optimizer)

            plotter.plot('loss', 'train', 'Loss train', j, loss_train.item(), xlabel='epoch')
            plotter.plot('loss', 'eval', 'Loss eval', j, loss_eval.item(), xlabel='epoch')
            for acc in downstream_accs:
                plotter.plot('acc', f'accuracy {acc[1]}', 'Downstream accuracy', j, acc[0], xlabel='epoch')
            self.save_metrics(j, loss_train, loss_eval, downstream_accs, downstream_datasets)

    def save_metrics(self, epoch, loss_train, loss_eval, donwstream_accs, downstream_datasets):
        metrics_file = self.checkpoint_dir / self.metrics_filename
        row = f'epoch: {epoch}, loss_train: {loss_train}, loss_eval: {loss_eval}, downstream accuracy '
        for i, dataset in enumerate(downstream_datasets):
            row += f'{dataset}: {donwstream_accs[i]},'
        row += '\n'
        with open(metrics_file, 'a') as f:
            f.write(row)

    def predict(self, texts, batch_size=64):
        self.qt.eval()
        eval_corpus = Corpus(texts, self.vocab, no_zeros=True)
        if len(eval_corpus) < batch_size:
            batch_size = len(eval_corpus)
        eval_iter = DataLoader(eval_corpus,
                               batch_size=batch_size,
                               num_workers=1,
                               drop_last=False,
                               pin_memory=True,  # send to GPU
                               collate_fn=safe_pack_sequence)
        eval_iter_tq = tqdm(eval_iter)
        predictions = None
        for i, batch_data in enumerate(eval_iter_tq):
            prediction = self.forward_pass(self.qt, batch_data, mode='eval', return_only_embedding=True)
            if predictions is None:
                predictions = prediction
            else:
                predictions = np.vstack((predictions, prediction))
        return predictions

    @classmethod
    def create_from_conf(cls, config_path):
        import importlib.machinery
        CONFIG = importlib.machinery.SourceFileLoader('CONFIG', config_path).load_module().CONFIG
        checkpoint_dir = Path(CONFIG['checkpoint_dir'])
        checkpoint_dir.mkdir()
        shutil.copyfile(config_path, checkpoint_dir / 'config.py')
        return cls(CONFIG['checkpoint_dir'], CONFIG['embedding'], CONFIG['data_path'], CONFIG['batch_size'],
                   CONFIG['hidden_size'], CONFIG['lr'], CONFIG['resume'], CONFIG['num_epochs'],
                   CONFIG['norm_threshold'], CONFIG['optimiser_class'])

    @classmethod
    def create_from_checkpoint(cls, checkpoint_dir, checkpoint_file_name='checkpoint_latest.pth',
                               config_file_name='config.py'):
        import importlib.machinery
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / config_file_name
        CONFIG = importlib.machinery.SourceFileLoader('CONFIG', str(config_path)).load_module().CONFIG
        learner = cls(checkpoint_dir, CONFIG['embedding'], CONFIG['data_path'], CONFIG['batch_size'],
                      CONFIG['hidden_size'], CONFIG['lr'], CONFIG['resume'], CONFIG['num_epochs'],
                      CONFIG['norm_threshold'], optimiser_class=CONFIG['optimiser_class'])

        checkpoint_states = torch.load(checkpoint_dir / checkpoint_file_name)
        learner.qt.load_state_dict(checkpoint_states['state_dict'])
        learner.optimizer.load_state_dict(checkpoint_states['optimizer'])

        return learner
