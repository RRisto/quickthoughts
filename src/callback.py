class Callback():
    def begin_fit(self):
        return True

    def after_fit(self): return True

    def begin_epoch(self, epoch):
        self.epoch = epoch
        return True

    def begin_validate(self): return True

    def after_epoch(self): return True

    def begin_batch(self, data):
        self.data = data
        return True

    def after_loss(self, loss):
        self.loss = loss
        return True

    def after_backward(self): return True

    def after_step(self): return True


class CallbackHandler():
    def __init__(self, cbs=None):
        self.cbs = cbs if cbs else []

    def set_learn(self, learn):
        self.learn = learn
        for cb in self.cbs:
            cb.learn = self.learn

    def add_callback(self, cb):
        cb.learn = self.learn
        self.cbs.append(cb)

    def begin_fit(self):
        self.in_train = True
        self.learn.stop = False
        res = True
        for cb in self.cbs: res = res and cb.begin_fit()
        return res

    def after_fit(self):
        res = not self.in_train
        for cb in self.cbs: res = res and cb.after_fit()
        return res

    def begin_epoch(self, epoch):
        self.learn.model.train()
        self.in_train = True
        res = True
        for cb in self.cbs: res = res and cb.begin_epoch(epoch)
        return res

    def begin_validate(self):
        self.learn.model.eval()
        self.in_train = False
        res = True
        for cb in self.cbs: res = res and cb.begin_validate()
        return res

    def after_epoch(self):
        res = True
        for cb in self.cbs: res = res and cb.after_epoch()
        return res

    def begin_batch(self, data):
        res = True
        for cb in self.cbs: res = res and cb.begin_batch(data)
        return res

    def after_loss(self, loss):
        res = self.in_train
        for cb in self.cbs:
            res = res and cb.after_loss(loss)
        return res

    def after_backward(self):
        res = True
        for cb in self.cbs: res = res and cb.after_backward()
        return res

    def after_step(self):
        res = True
        for cb in self.cbs: res = res and cb.after_step()
        return res

    def do_stop(self):
        try:
            return self.learn.stop
        finally:
            self.learn.stop = False
