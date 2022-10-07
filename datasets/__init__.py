from ml4vision.ml.datasets import SegmentationDataset

def get_dataset(name, dataset_kwargs={}):
    if name == "ml4vision":
        return SegmentationDataset(**dataset_kwargs)
    else:
        raise RuntimeError(f'Dataset {name} is not available!') 