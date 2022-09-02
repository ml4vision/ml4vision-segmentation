from ml4vision.ml.datasets import SegmentationDataset as RemoteSegmentationDataset
from .LocalSegmentationDataset import LocalSegmentationDataset

def get_dataset(name, dataset_kwargs={}):
    if name == "remote":
        return RemoteSegmentationDataset(**dataset_kwargs)
    elif name == "local":
        return LocalSegmentationDataset(**dataset_kwargs)
    else:
        raise RuntimeError(f'Dataset {name} is not available!') 