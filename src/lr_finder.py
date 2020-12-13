import numpy as np
import matplotlib.pyplot as plt
from src.callback_base import Callback
from src.sched import annealing_linear, Scheduler


class LRFinder(Callback):
    "Causes `learn` to go on a mock training from `start_lr` to `end_lr` for `num_it` iterations."

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, num_it: int = 4, stop_div: bool = True,
                 annealing_func=annealing_linear, beta=.98):
        self.stop_div = stop_div
        self.sched = Scheduler((start_lr, end_lr), num_it, annealing_func)
        self.lrs = []
        self.losses = []
        self.beta = beta
        self.avg_loss = 0

    def begin_fit(self):
        "Initialize optimizer and learner hyperparameters."
        self.opt = self.learn.opt
        self.opt.lr = self.sched.start
        self.stop, self.best_loss = False, 0.
        self.best_lr = 0
        self.iteration = 0
        return True

    def after_loss(self, loss):
        if self.handler.in_train:
            "Determine if loss has runaway and we should stop."
            self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss.item()
            smooth_loss = self.avg_loss / (1 - self.beta ** (self.iteration + 1))

            if self.iteration == 0 or smooth_loss < self.best_loss:
                self.best_loss = smooth_loss
                self.best_lr = self.opt.lr
            self.lrs.append(self.opt.lr)
            self.losses.append(smooth_loss)
            self.iteration += 1
            if self.sched.is_done or (self.stop_div and (smooth_loss > 4 * self.best_loss or np.isnan(smooth_loss))):
                # We use the smoothed loss to decide on the stopping since it's less shaky.
                if not self.stop:
                    self.stop = self.iteration
                    self.learn.stop = True
                return False
            return True
        return True

    def after_step(self):
        self.opt.lr = self.sched.step()
        return True

    def begin_validate(self):
        plt.plot(self.lrs[5:-5], self.losses[5:-5])
        plt.xscale('log')
        plt.show()
        print(f'Best loss {round(self.best_loss, 3)}')
        print(f'Best lr {round(self.best_lr, 3)}')
        self.learn.stop = True  # will set cb_handler do_stop() True no reason to continue
        return False
