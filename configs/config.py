from yacs.config import CfgNode as CN

def get_config(project_location='./data', categories=None):

    # define defaults
    config = CN()

    config.categories = categories

    config.save = True
    config.save_location = './output'

    config.display = True
    config.display_it = 50

    config.pretrained_model = None

    config.train_dataset = CN(dict(
        name = 'ml4vision', 
        params = CN(dict(
            location = project_location,
            split = 'TRAIN',
            fake_size = 500,
            ignore_zero = False if len(categories) == 1 else True
        )),
        batch_size = 4,
        num_workers = 4
    ))

    config.val_dataset = CN(dict(
        name = 'ml4vision',
        params = CN(dict(
            location = project_location,
            split = 'VAL',
            ignore_zero = False if len(categories) == 1 else True
        )),
        batch_size = 1,
        num_workers = 4
    ))

    config.model = CN(dict(
        name = 'unet',
        params = CN(dict(
            encoder_name = 'resnet18',
            classes = len(categories)
        ))
    ))

    config.loss = CN(dict(
        name = 'bcedice' if len(categories) == 1 else 'cedice',
        params = CN(dict(
            ignore_index = 255
        ))
    ))

    config.solver = CN(dict(
        lr = 5e-4,
        patience = 3 # number of epochs with no improvement after which learning rate will be reduced
    ))

    config.transform = CN(dict(
        resize = True,
        min_size = 512,
        random_crop = True,
        crop_size = 256,
        scale = 0,
        flip_horizontal = True,
        flip_vertical = True,
        random_brightness_contrast = True
    ))

    return config
