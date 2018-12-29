class AverageMeter(object):
    """Modified from https://github.com/Cysu/open-reid.
    Computes and stores the average and current value"""

    def __init__(self, name='', fmt='{:.4f}'):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.fmt = fmt

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-20)

    @property
    def val_str(self):
        return self.name + ' ' + self.fmt.format(self.val)

    @property
    def avg_str(self):
        return self.name + ' ' + self.fmt.format(self.avg)


class RunningAverageMeter(object):
    """Computes and stores the running average and current value"""

    def __init__(self, hist=0.99, name='', fmt='{:.4f}'):
        self.val = None
        self.avg = None
        self.hist = hist
        self.name = name
        self.fmt = fmt

    def reset(self):
        self.val = None
        self.avg = None

    def update(self, val):
        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.avg * self.hist + val * (1 - self.hist)
        self.val = val

    @property
    def val_str(self):
        return self.name + ' ' + self.fmt.format(self.val)

    @property
    def avg_str(self):
        return self.name + ' ' + self.fmt.format(self.avg)


class RecentAverageMeter(object):
    """Stores and computes the average of recent values."""

    def __init__(self, hist_size=100, name='', fmt='{:.4f}'):
        self.hist_size = hist_size
        self.fifo = []
        self.val = 0
        self.name = name
        self.fmt = fmt

    def reset(self):
        self.fifo = []
        self.val = 0

    def update(self, val):
        self.val = val
        self.fifo.append(val)
        if len(self.fifo) > self.hist_size:
            del self.fifo[0]

    @property
    def avg(self):
        assert len(self.fifo) > 0
        return float(sum(self.fifo)) / len(self.fifo)

    @property
    def val_str(self):
        return self.name + ' ' + self.fmt.format(self.val)

    @property
    def avg_str(self):
        return self.name + ' ' + self.fmt.format(self.avg)