"""Refer to `torch.utils.data.dataloader`, there is a `_DataLoaderIter` and a
`DataLoader` class. To combine multi dataloaders for multi-task learning, and
to mimic the `for batch in dataloader:` syntax, we should also implement two
classes `_MTDataLoaderIter` and `MTDataLoader`."""

from __future__ import print_function


class _MTDataLoaderIter(object):
    """mt_loader: a MTDataLoader object"""
    def __init__(self, mt_loader):
        self.mt_loader = mt_loader
        self.loader_iters = [iter(loader) for loader in self.mt_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        if self.mt_loader.ref_shortest_loader:
            # When the shortest loader (the one with minimum number of batches)
            # terminates, this iterator will terminates.
            # The `StopIteration` raised inside that shortest loader's `__next__`
            # method will in turn gets out of this `__next__` method.
            batches = [loader_iter.next() for loader_iter in self.loader_iters]
        else:
            batches = []
            for idx, loader_iter in enumerate(self.loader_iters):
                try:
                    batch = loader_iter.next()
                except StopIteration:
                    # If it's the specified loader terminates, we terminate
                    if idx == self.mt_loader.ref_loader_idx:
                        raise StopIteration
                    # Otherwise, start a new epoch for the current loader
                    self.loader_iters[idx] = iter(self.mt_loader.loaders[idx])
                    batch = self.loader_iters[idx].next()
                batches.append(batch)
        return self.mt_loader.combine_batch(batches)

    # Python 2 compatibility
    next = __next__

    def __len__(self):
        return len(self.mt_loader)


class MTDataLoader(object):
    """Different dataloaders have different lengths, the length of the multi-task
    loader (i.e. when to terminate) can be defined by the shortest loader, or by
    the specified loader.
    Args:
        loaders: a list/tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders, ref_shortest_loader=False, ref_loader_idx=0):
        assert ref_loader_idx in range(len(loaders))
        self.loaders = loaders
        self.ref_shortest_loader = ref_shortest_loader
        self.ref_loader_idx = ref_loader_idx

    def __iter__(self):
        return _MTDataLoaderIter(self)

    def __len__(self):
        lengths = [len(loader) for loader in self.loaders]
        return min(lengths) if self.ref_shortest_loader else lengths[self.ref_loader_idx]

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches


if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    loader1 = DataLoader(
        torchvision.datasets.FakeData(
            size=100,
            image_size=(3, 8, 8),
            num_classes=10,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        ),
        batch_size=4,
        shuffle=True
    )

    loader2 = DataLoader(
        torchvision.datasets.FakeData(
            size=80,
            image_size=(3, 8, 8),
            num_classes=10,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        ),
        batch_size=8,
        shuffle=True
    )

    loader3 = DataLoader(
        torchvision.datasets.FakeData(
            size=40,
            image_size=(3, 8, 8),
            num_classes=10,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        ),
        batch_size=2,
        shuffle=True
    )

    def loop_through_mt_loader(loader):
        """A use case of iterating through mt_loader."""
        for i, batches in enumerate(loader):
            # print some info
            if i in [0, 1]:
                for j, b in enumerate(batches):
                    data, target = b
                    print('MT Batch Idx {}, Sub Batch Idx {}, Data Type {}, Size {}, Target Type {}, Size {}'.format(i, j, data.dtype, data.size(), target.dtype, target.size()))
        print('NO. MT Batches At Termination: {}'.format(i + 1))

    print('{}\n MTDataLoader Length Refer to Shortest Loader\n{}'.format('*' * 80, '*' * 80))
    mt_loader = MTDataLoader([loader1, loader2, loader3], ref_shortest_loader=True)
    for i in range(3):
        print('{}\nLoop Through MTDataLoader #{}\n{}'.format('-' * 40, i + 1, '-' * 40))
        loop_through_mt_loader(mt_loader)
    print('Each Loader Length:', ', '.join([str(len(l)) for l in mt_loader.loaders]))
    print('MTDataLoader Length:', len(mt_loader))

    ref_loader_idx = 0
    print('{}\n MTDataLoader Length Refer to Loader idx {} \n{}'.format('*' * 80, ref_loader_idx, '*' * 80))
    mt_loader = MTDataLoader([loader1, loader2, loader3], ref_shortest_loader=False, ref_loader_idx=ref_loader_idx)
    for i in range(3):
        print('{}\nLoop Through MTDataLoader #{}\n{}'.format('-' * 40, i + 1, '-' * 40))
        loop_through_mt_loader(mt_loader)
    print('Each Loader Length:', ', '.join([str(len(l)) for l in mt_loader.loaders]))
    print('MTDataLoader Length:', len(mt_loader))

    ref_loader_idx = 2
    print('{}\n MTDataLoader Length Refer to Loader idx {} \n{}'.format('*' * 80, ref_loader_idx, '*' * 80))
    mt_loader = MTDataLoader([loader1, loader2, loader3], ref_shortest_loader=False, ref_loader_idx=ref_loader_idx)
    for i in range(3):
        print('{}\nLoop Through MTDataLoader #{}\n{}'.format('-' * 40, i + 1, '-' * 40))
        loop_through_mt_loader(mt_loader)
    print('Each Loader Length:', ', '.join([str(len(l)) for l in mt_loader.loaders]))
    print('MTDataLoader Length:', len(mt_loader))
