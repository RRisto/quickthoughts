from src.callback_base import Callback


class OneCycle(Callback):

    def begin_fit(self):
        self.base_lr = self.learn.lr
        self.lrs = []
        self.num = 0
        return True

    def begin_batch(self, data):
        if not self.handler.in_train:
            return
        self.num += 1
        n = len(self.learn.train_iter)
        bn = self.epoch * n + self.num
        mn = self.learn.num_epochs * n
        pct = bn / mn
        pct_start, div_start = 0.25, 10
        if pct < pct_start:
            pct /= pct_start
            lr = (1 - pct) * self.base_lr / div_start + pct * self.base_lr
        else:
            pct = (pct - pct_start) / (1 - pct_start)
            lr = (1 - pct) * self.base_lr
        self.learn.opt.lr = lr
        self.lrs.append(lr)
