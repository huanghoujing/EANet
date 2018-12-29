from resnet import get_resnet


backbone_factory = {
    'resnet18': get_resnet,
    'resnet34': get_resnet,
    'resnet50': get_resnet,
    'resnet101': get_resnet,
    'resnet152': get_resnet,
}


def create_backbone(cfg):
    return backbone_factory[cfg.name](cfg)
