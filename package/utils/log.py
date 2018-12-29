import sys
import os
import os.path as osp
import datetime
from .file import may_make_dir


class ReDirectSTD(object):
    """Modified from Tong Xiao's `Logger` in open-reid.
    This class overwrites sys.stdout or sys.stderr, so that console logs can
    also be written to file.
    Args:
      fpath: file path
      console: one of ['stdout', 'stderr']
      immediately_visible: If `False`, the file is opened only once and closed
        after exiting. In this case, the message written to file may not be
        immediately visible (Because the file handle is occupied by the
        program?). If `True`, each writing operation of the console will
        open, write to, and close the file. If your program has tons of writing
        operations, the cost of opening and closing file may be obvious. (?)
    Usage example:
      `ReDirectSTD('stdout.txt', 'stdout', False)`
      `ReDirectSTD('stderr.txt', 'stderr', False)`
    NOTE: File will be deleted if already existing. Log dir and file is created
      lazily -- if no message is written, the dir and file will not be created.
    """

    def __init__(self, fpath=None, console='stdout', immediately_visible=False):
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == 'stdout' else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visible = immediately_visible
        if fpath is not None:
            # Remove existing log file.
            if osp.exists(fpath):
                os.remove(fpath)

        # Overwrite
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            may_make_dir(os.path.dirname(osp.abspath(self.file)))
            if self.immediately_visible:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)
        self.flush()

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H-%M-%S'
    return datetime.datetime.today().strftime(fmt)


def array_str(array, fmt='{:.2f}', sep=', ', with_boundary=True):
    """String of a 1-D tuple, list, or numpy array containing digits."""
    ret = sep.join([fmt.format(float(x)) for x in array])
    if with_boundary:
        ret = '[' + ret + ']'
    return ret


def array_2d_str(array, fmt='{:.2f}', sep=', ', row_sep='\n', with_boundary=True):
    """String of a 2-D tuple, list, or numpy array containing digits."""
    ret = row_sep.join([array_str(x, fmt=fmt, sep=sep, with_boundary=with_boundary) for x in array])
    if with_boundary:
        ret = '[' + ret + ']'
    return ret


def tight_float_str(x, fmt='{:.4f}'):
    return fmt.format(x).rstrip('0').rstrip('.')


def score_str(x):
    return '{:5.1%}'.format(x).rjust(6)


def join_str(sequence, sep):
    sequence = [s for s in sequence if s != '']
    return sep.join(sequence)


def write_to_file(file, msg, append=True):
    with open(file, 'a' if append else 'w') as f:
        f.write(msg)
