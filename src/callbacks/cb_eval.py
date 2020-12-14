import numpy as np
from src.callback_base import Callback


class EvalSaveMetrics(Callback):

    def __init__(self, downstream_evaluation_func=None, downstream_eval_datasets=None, plotter=None):
        self.downstream_evaluation_func = downstream_evaluation_func
        self.downstream_eval_datasets = downstream_eval_datasets
        self.plotter = plotter
        self.losses_train = []
        self.losses_val = []
        self.n_epochs = 0

    def begin_epoch(self, epoch):
        self.losses_train = []
        self.losses_val = []
        self.n_epochs += 1
        return True

    def after_loss(self, loss):
        if self.handler.in_train:
            self.losses_train.append(loss.item())
        else:
            self.losses_val.append(loss.item())
        return True

    def plot_metrics(self, downstream_accs):
        self.plotter.plot('loss', 'train', 'Loss train', self.n_epochs, np.mean(self.losses_train), xlabel='epoch')
        self.plotter.plot('loss', 'eval', 'Loss eval', self.n_epochs, np.mean(self.losses_val), xlabel='epoch')
        if downstream_accs is not None:
            for acc in downstream_accs:
                self.plotter.plot('acc', f'accuracy {acc[1]}', 'Downstream accuracy', self.n_epochs, acc[0],
                                  xlabel='epoch')

    def after_epoch(self):
        log_row = f'epoch: , loss_train: {np.mean(self.losses_train)}, loss_eval: {np.mean(self.losses_val)}, ' \
            f'downstream accuracy '

        downstream_accs = None
        if self.downstream_evaluation_func is not None and self.downstream_eval_datasets is not None:
            log_row, downstream_accs = self.evaluate_downstream_tasks(log_row)
        if self.plotter is not None:
            self.plot_metrics(downstream_accs)

        log_row += '\n'

        self.metrics_file = self.learn.checkpoint_dir / self.learn.metrics_filename
        with open(self.metrics_file, 'a') as f:
            f.write(log_row)
        return True

    def evaluate_downstream_tasks(self, log_row):
        self.learn.model.eval()
        downstream_accs = self.downstream_evaluation_func(self.learn.predict, datasets=self.downstream_eval_datasets)
        print(f'validation accuracies {downstream_accs}')
        for i, dataset in enumerate(self.downstream_eval_datasets):
            log_row += f'{dataset}: {downstream_accs[i]},'
        self.learn.model.train()
        return log_row, downstream_accs
