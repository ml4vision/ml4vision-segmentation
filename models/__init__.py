import torch
from segmentation_models_pytorch import create_model


def get_model(name, model_kwargs={}):
    model = create_model(name, **model_kwargs)
    return model