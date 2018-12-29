from collections import OrderedDict


class Loss(object):
    """Base class for calculating loss and managing log.
    """
    def __init__(self, cfg, tb_writer=None):
        self.cfg = cfg
        self.meter_dict = OrderedDict()
        self.tb_writer = tb_writer

    def reset_meters(self):
        for k, v in self.meter_dict.items():
            v.reset()

    def __call__(self, batch, pred, step=0, **kwargs):
        """Return a dict, whose keys at least include ['loss']. In some cases, e.g.
        for saving GPU memory, you may perform backward in this function. In this case,
        the returned loss should be a python scalar, so that the outer logic will not
        perform backward."""
        pass
