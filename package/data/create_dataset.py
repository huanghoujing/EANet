from .datasets.market1501 import Market1501
from .datasets.cuhk03_np_detected_png import CUHK03NpDetectedPng
from .datasets.cuhk03_np_detected_jpg import CUHK03NpDetectedJpg
from .datasets.duke import DukeMTMCreID
from .datasets.coco import COCO
from .datasets.msmt17 import MSMT17
from .datasets.partial_reid import PartialREID
from .datasets.partial_ilids import PartialiLIDs


__factory = {
        'market1501': Market1501,
        'cuhk03_np_detected_png': CUHK03NpDetectedPng,
        'cuhk03_np_detected_jpg': CUHK03NpDetectedJpg,
        'duke': DukeMTMCreID,
        'coco': COCO,
        'msmt17': MSMT17,
        'partial_reid': PartialREID,
        'partial_ilids': PartialiLIDs,
    }


dataset_shortcut = {
    'market1501': 'M',
    'cuhk03_np_detected_png': 'C',
    'cuhk03_np_detected_jpg': 'C',
    'duke': 'D',
    'msmt17': 'MS',
    'partial_reid': 'PR',
    'partial_ilids': 'PI',
}


def create_dataset(cfg, samples=None):
    return __factory[cfg.name](cfg, samples=samples)
