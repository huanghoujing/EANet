def str2bool(v):
    """From https://github.com/amdegroot/ssd.pytorch"""
    return v.lower() in ("yes", "true", "t", "1")


class CommaSeparatedSeq(object):
    def __init__(self, seq_class=tuple, func=int):
        self.seq_class = seq_class
        self.func = func

    def __call__(self, s):
        return self.seq_class([self.func(i) for i in s.split(',')])
