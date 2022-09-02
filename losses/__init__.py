from torch import nn
from .losses import BCEDiceLoss, CEDiceLoss


def get_loss(name, loss_kwargs={}):
    if name == 'bce':
        return nn.BCEWithLogitsLoss(**loss_kwargs)
    elif name == 'crossentropy':
        return nn.CrossEntropyLoss(**loss_kwargs)
    elif name == 'bcedice':
        return BCEDiceLoss(**loss_kwargs)
    elif name == 'cedice':
        return CEDiceLoss(**loss_kwargs)
    else:
        raise RuntimeError(f'Loss {name} is not available!')