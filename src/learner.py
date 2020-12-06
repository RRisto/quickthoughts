import logging
import shutil

import torch
from pathlib import Path

from gensim.utils import tokenize
from tqdm import tqdm
from pprint import pformat
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.callback import CallbackHandler
from src.lr_finder import LRFinder
from src.sched import annealing_exp
from src.utils import load_pretrained_embeddings
from src.qt_model import QuickThoughts
from src.utils import checkpoint_training, safe_pack_sequence, VisdomLinePlotter
from src.data.corpus import create_train_eval_corpus, Corpus

_LOGGER = logging.getLogger(__name__)


class QTLearner:
    def __init__(self, checkpoint_dir, embedding, data_path, seq_max_len, batch_size, hidden_size, lr, num_epochs,
                 norm_threshold, emb_dim, test_downstream_task_func, test_downstream_datasets, tokenizer_func=tokenize,
                 config_file_name='config.json', optimizer_class=optim.Adam, metrics_filename='metrics.txt',
                 eval_p=0.2, cb=[], cuda=False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_file_name = config_file_name
        self.metrics_filename = metrics_filename
        self.embedding = embedding
        self.data_path = data_path
        self.seq_max_len = seq_max_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.norm_threshold = norm_threshold
        self.emb_dim = emb_dim
        self.optimizer_class = optimizer_class
        self.tokenizer_func = tokenizer_func
        self.train_iter, self.eval_iter, self.stoi = self.create_dataloaders(eval_p)
        self.WV_MODEL = load_pretrained_embeddings(self.embedding, self.stoi)
        # model, optimizer, and loss function
        self.cuda = cuda
        self.device = 'cuda' if self.cuda else 'cpu'
        self.model = QuickThoughts(self.WV_MODEL, self.stoi, self.hidden_size, emb_dim=self.emb_dim,
                                   device=self.device).to(self.device)
        self.opt = self.optimizer_class(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.test_downstream_task_func = test_downstream_task_func
        self.test_downstream_task_datasets = test_downstream_datasets
        self.cbs = CallbackHandler(cb)
        self.cbs.set_learn(self)

    def gather_conf_info(self):
        return {'checkpoint_dir': str(self.checkpoint_dir),
                'embedding': self.embedding,
                'data_path': self.data_path,
                'batch_size': self.batch_size,
                'hidden_size': self.hidden_size,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'norm_threshold': self.norm_threshold,
                'optimizer_class': self.optimizer_class
                }

    def _init_logging(self):
        _LOGGER.info(pformat(self.gather_conf_info()))

    def lr_find(self, start_lr=1e-7, end_lr=10, num_it: int = 49, stop_div: bool = True, wd: float = None,
                annealing_func=annealing_exp):
        "Explore lr from `start_lr` to `end_lr` over `num_it` iterations in `learn`. If `stop_div`, stops when loss diverges."
        cb = LRFinder(start_lr, end_lr, num_it, stop_div, annealing_func=annealing_func)
        self.cbs.add_callback(cb)
        epochs = int(np.ceil(num_it / len(self.train_iter)))
        self.fit(False)

    def forward_pass(self, data, mode='train', return_only_embedding=False):
        # forward pass
        if mode == 'train':
            enc_f, enc_g = self.model(data)
        else:  # in eval mode returns concatenated enc_f and enc_g
            enc = self.model(data)
            if return_only_embedding:
                return enc.detach().cpu().numpy()
            enc_f, enc_g = enc[:, :self.hidden_size], enc[:, self.hidden_size:]

        # calculate scores
        scores = torch.matmul(enc_f, enc_g.t())

        # zero out when it's the same sentence
        if torch.cuda.is_available() and self.cuda:
            mask = torch.eye(len(scores)).cuda().bool()
        else:
            mask = torch.eye(len(scores)).bool()

        scores.masked_fill_(mask, 0)

        # return log scores and target
        block_log_scores = F.log_softmax(scores, dim=1)
        # targets also topelitz matrix
        targets = self.model.generate_targets(self.batch_size, offsetlist=[1])
        loss = self.kl_loss(block_log_scores, targets)
        return loss, block_log_scores

    def fit_batch(self, data, mode='train'):
        if not self.cbs.begin_batch(data):
            return

        loss, block_log_scores = self.forward_pass(data, mode)
        if mode == 'train':
            if not self.cbs.after_loss(loss):
                return None, None
            loss.backward()
            # grad clipping todo make callback
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.norm_threshold)
            # optimizer.step()
            if self.cbs.after_backward():
                self.cbs.learn.opt.step()
            if self.cbs.after_step():
                self.cbs.learn.opt.zero_grad()

        return loss, block_log_scores

    def fit_eval_epoch(self, train_data_iter, eval_data_iter, failed_or_skipped_batches):
        loss_train, failed_or_skipped_batches_train = self.fit_epoch(train_data_iter, failed_or_skipped_batches,
                                                                     mode='train')
        failed_or_skipped_batches += failed_or_skipped_batches_train
        loss_eval, _ = self.fit_epoch(eval_data_iter, failed_or_skipped_batches, mode='eval')
        return loss_train, loss_eval, failed_or_skipped_batches

    def fit_epoch(self, data_iter, failed_or_skipped_batches, mode='train'):

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_iter_tq = tqdm(data_iter)
        loss = 0
        for i, data in enumerate(data_iter_tq):
            # deal with bad sequences
            if not data:
                failed_or_skipped_batches += 1
                continue

            # zero out
            try:
                if torch.cuda.is_available() and self.cuda:
                    torch.cuda.empty_cache()
                # optimizer.zero_grad()
                if torch.cuda.is_available() and self.cuda:
                    data = data.cuda()

                loss, _ = self.fit_batch(data, mode)

                data_iter_tq.set_description(
                    "loss {} {:.4f} | failed/skipped {:3d}".format(mode, loss, failed_or_skipped_batches))

            except Exception as e:
                _LOGGER.exception(e)
                failed_or_skipped_batches += 1
                if torch.cuda.is_available() and self.cuda:
                    torch.cuda.empty_cache()

            if self.cbs.do_stop():
                return None, None
        return loss, failed_or_skipped_batches

    def eval_downstream(self, qt, loc='data'):
        qt.eval()
        accs = self.test_downstream_task_func(self.predict, datasets=self.test_downstream_task_datasets, loc=loc)
        return accs

    def create_dataloaders(self, eval_p=0.2):
        bookcorpus_train, bookcorpus_eval, stoi = create_train_eval_corpus(self.data_path, self.tokenizer_func, eval_p,
                                                                           self.seq_max_len)
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
        return train_iter, eval_iter, stoi

    def fit(self, plot=False):
        if not self.cbs.begin_fit():
            return

        self._init_logging()
        if plot:
            plotter = VisdomLinePlotter()

        # start training
        self.model = self.model.train()
        failed_or_skipped_batches = 0
        best_eval_loss = None
        for j in range(self.num_epochs):
            if not self.cbs.begin_epoch(j):
                continue

            loss_train, loss_eval, failed_or_skipped_batches = self.fit_eval_epoch(self.train_iter, self.eval_iter,
                                                                                   failed_or_skipped_batches)

            if self.cbs.begin_validate():  # todo maybe use other callback or pack it into callback
                downstream_accs = self.eval_downstream(self.model)
                if best_eval_loss is None or best_eval_loss > loss_eval:
                    best_eval_loss = loss_eval
                    checkpoint_training(self.checkpoint_dir, j, self.model, self.opt)

                if plot:
                    plotter.plot('loss', 'train', 'Loss train', j, loss_train.item(), xlabel='epoch')
                    plotter.plot('loss', 'eval', 'Loss eval', j, loss_eval.item(), xlabel='epoch')
                    for acc in downstream_accs:
                        plotter.plot('acc', f'accuracy {acc[1]}', 'Downstream accuracy', j, acc[0], xlabel='epoch')
                self.save_metrics(j, loss_train, loss_eval, downstream_accs, self.test_downstream_task_datasets)

            if self.cbs.do_stop() or not self.cbs.after_epoch():
                break

        self.cbs.after_fit()

    def save_metrics(self, epoch, loss_train, loss_eval, donwstream_accs, downstream_datasets):
        metrics_file = self.checkpoint_dir / self.metrics_filename
        row = f'epoch: {epoch}, loss_train: {loss_train}, loss_eval: {loss_eval}, downstream accuracy '
        for i, dataset in enumerate(downstream_datasets):
            row += f'{dataset}: {donwstream_accs[i]},'
        row += '\n'
        with open(metrics_file, 'a') as f:
            f.write(row)

    def predict(self, texts, batch_size=64):
        self.model.eval()
        eval_corpus = Corpus(texts, self.stoi, tokenizer_func=self.tokenizer_func, no_zeros=True)
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
            prediction = self.forward_pass(batch_data, mode='eval', return_only_embedding=True)
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
        return cls(checkpoint_dir=CONFIG['checkpoint_dir'],
                   embedding=CONFIG['embedding'],
                   data_path=CONFIG['data_path'],
                   seq_max_len=CONFIG['seq_max_len'],
                   batch_size=CONFIG['batch_size'],
                   hidden_size=CONFIG['hidden_size'],
                   lr=CONFIG['lr'],
                   num_epochs=CONFIG['num_epochs'],
                   norm_threshold=CONFIG['norm_threshold'],
                   emb_dim=CONFIG['emb_dim'],
                   test_downstream_task_func=CONFIG['downstream_evaluation_func'],
                   test_downstream_datasets=CONFIG['downstream_eval_datasets'],
                   tokenizer_func=CONFIG['tokenizer_func'],
                   optimizer_class=CONFIG['optimiser_class'],
                   eval_p=CONFIG['eval_p'],
                   cb=CONFIG['cb'],
                   cuda=CONFIG['cuda']
                   )

    @classmethod
    def create_from_checkpoint(cls, checkpoint_dir, checkpoint_file_name='checkpoint_latest.pth',
                               config_file_name='config.py'):
        import importlib.machinery
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / config_file_name
        CONFIG = importlib.machinery.SourceFileLoader('CONFIG', str(config_path)).load_module().CONFIG
        learner = cls(checkpoint_dir=checkpoint_dir,
                      embedding=CONFIG['embedding'],
                      data_path=CONFIG['data_path'],
                      seq_max_len=CONFIG['seq_max_len'],
                      batch_size=CONFIG['batch_size'],
                      hidden_size=CONFIG['hidden_size'], lr=CONFIG['lr'],
                      num_epochs=CONFIG['num_epochs'],
                      norm_threshold=CONFIG['norm_threshold'],
                      emb_dim=CONFIG['emb_dim'],
                      test_downstream_task_func=CONFIG['downstream_evaluation_func'],
                      test_downstream_datasets=CONFIG['downstream_eval_datasets'],
                      tokenizer_func=CONFIG['tokenizer_func'],
                      optimizer_class=CONFIG['optimiser_class'],
                      eval_p=CONFIG['eval_p'],
                      cb=CONFIG['cb'],
                      cuda=CONFIG['cuda']
                      )

        checkpoint_states = torch.load(checkpoint_dir / checkpoint_file_name)
        learner.model.load_state_dict(checkpoint_states['state_dict'])
        learner.opt.load_state_dict(checkpoint_states['optimizer'])

        return learner
