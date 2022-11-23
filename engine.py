import os
import shutil
import torch
from datasets import get_dataset
from models import get_model
from losses import get_loss
from utils.meters import AverageMeter
from utils.transforms import get_train_transform, get_val_transform
from utils.iou_evaluator import IOUEvaluator
from utils.visualizer import SegmentationVisualizer
from tqdm import tqdm
from ml4vision.client import MLModel

class Engine:

    def __init__(self, config, device=None):

        self.config = config
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create output dir
        if config.save:
            os.makedirs(config.save_location, exist_ok=True)

        # train/val dataloaders
        self.train_dataset_it, self.val_dataset_it = self.get_dataloaders(config, self.device)
        
        # model
        self.model = self.get_model(config, self.device)
        
        # loss
        self.loss_fn = self.get_loss(config)
        
        # optimizer/scheduler
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(config, self.model)

        # visualizer 
        self.visualizer = SegmentationVisualizer()

    @staticmethod
    def get_dataloaders(config, device):
        train_transform = get_train_transform(config.transform)
        train_dataset = get_dataset(
            config.train_dataset.name, config.train_dataset.params)
        train_dataset.transform = train_transform
        train_dataset_it = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_dataset.batch_size, shuffle=True, drop_last=True, num_workers=config.train_dataset.num_workers, pin_memory=True if device.type == 'cuda' else False)

        # val dataloader
        val_transform = get_val_transform(config.transform)
        val_dataset = get_dataset(
            config.val_dataset.name, config.val_dataset.params)
        val_dataset.transform = val_transform
        val_dataset_it = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.val_dataset.batch_size, shuffle=False, drop_last=False, num_workers=config.val_dataset.num_workers, pin_memory=True if device.type == 'cuda' else False)

        return train_dataset_it, val_dataset_it

    @staticmethod
    def get_model(config, device):
        model = get_model(config.model.name, config.model.params).to(device)

        # load checkpoint
        if config.pretrained_model is not None and os.path.exists(config.pretrained_model):
            print(f'Loading model from {config.pretrained_model}')
            state = torch.load(config.pretrained_model)
            model.load_state_dict(state['model_state_dict'], strict=True)

        return model

    @staticmethod
    def get_loss(config):

        loss_fn = get_loss(config.loss.name, config.loss.get('params'))

        return loss_fn

    @staticmethod
    def get_optimizer_and_scheduler(config, model):

        optimizer = torch.optim.Adam(model.parameters(), lr=config.solver.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=config.solver.patience, verbose=True)

        return optimizer, scheduler

    def display(self, pred, sample):
        # display a single image
        image = sample['image'][0]
        pred = (pred[0] > 0).cpu() if len(self.config.categories) == 1 else torch.argmax(pred[0],dim=0).cpu()
        gt = sample['label'][0]

        self.visualizer.display(image, pred, gt)

    def forward(self, sample):
        images = sample['image'].to(self.device)
        labels = sample['label'].to(self.device)

        if len(self.config.categories) == 1:
            labels = (labels > 0).unsqueeze(1).float()
        else:
            labels = labels.long()

        pred = self.model(images)
        loss = self.loss_fn(pred, labels)

        return pred, loss

    def train_step(self):
        config = self.config

        # define meters
        loss_meter = AverageMeter()
 
        self.model.train()
        for i, sample in enumerate(tqdm(self.train_dataset_it)):
            pred, loss = self.forward(sample)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())

            if config.display and i % config.display_it == 0:
                with torch.no_grad():
                    self.display(pred, sample)

        return loss_meter.avg

    def val_step(self):
        config = self.config

        # define meters
        loss_meter = AverageMeter()
        iou_meter = IOUEvaluator(len(config.categories) + 1 if len(config.categories) == 1 else len(config.categories))

        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_dataset_it)):
                pred, loss = self.forward(sample)
                loss_meter.update(loss.item())

                if config.display and i % config.display_it == 0:
                    self.display(pred, sample)

                # compute iou
                labels = sample['label'].to(self.device)
                labels = labels.unsqueeze(1).long()

                if len(config.categories) == 1: # binary
                    labels = (labels > 0).long()
                    iou_meter.addBatch((pred > 0).long(), labels)
                else:
                    iou_meter.addBatch(pred.argmax(dim=1,keepdim=True), labels)

        # get iou metric
        if len(config.categories) == 1:
            iou = iou_meter.getIoU()[1][1] 
            metrics = {'mean_iou': iou}
        else:
            miou, iou = iou_meter.getIoU()
            metrics = {'mean_iou': miou, 'class_iou': iou}

        return loss_meter.avg, metrics

    def save_checkpoint(self, epoch, is_best_val=False, best_val_loss=0, is_best_miou=False, best_miou=0, metrics={}):
        config = self.config

        state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }

        print("=> saving checkpoint")
        file_name = os.path.join(config.save_location, "checkpoint.pth")
        torch.save(state, file_name)
       
        if is_best_val:
            print("=> saving best_val checkpoint")
            shutil.copyfile(
                file_name, os.path.join(config.save_location, "best_val_model.pth")
            )

        if is_best_miou:
            print("=> saving best_miou checkpoint")
            shutil.copyfile(
                file_name, os.path.join(config.save_location, "best_miou_model.pth")
            )

    def train(self):  
        best_val_loss = float('inf')
        best_miou = 0

        # for epoch in range(config.solver.num_epochs):      
        epoch = 0
        while True:
            print(f'Starting epoch {epoch}')

            train_loss = self.train_step()
            val_loss, metrics = self.val_step()

            print(f'==> train loss: {train_loss}')
            print(f'==> val loss: {val_loss}')
            print(f'metrics: {metrics}')

            is_best_val = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)

            is_best_miou = metrics['mean_iou'] > best_miou
            best_miou = max(metrics['mean_iou'], best_miou)

            self.save_checkpoint(epoch, is_best_val=is_best_val, best_val_loss=best_val_loss, is_best_miou=is_best_miou, best_miou=best_miou, metrics=metrics)
            
            self.scheduler.step(val_loss)
            epoch = epoch + 1

            if self.optimizer.param_groups[0]['lr'] < self.config.solver.lr/100:
                break

    def upload(self, project):
        
        config = self.config

        # load best ap model
        model = self.get_model(config, torch.device('cpu'))
        state = torch.load(os.path.join(config.save_location, 'best_miou_model.pth'), map_location='cpu')
        model.load_state_dict(state['model_state_dict'], strict=True)
        model.eval()

        # trace model
        traced_model = torch.jit.trace(model, torch.randn(1, 3, config.transform.min_size, config.transform.min_size)) 
        traced_model.save(os.path.join(config.save_location, 'best_miou_model.pt'))

        # create model
        if project.model is None:
            model = MLModel.create(
                project.client,
                f'{project.name}-model',
                type='SEGMENTATION',
                project=project.uuid,
            )
        else:
            model = project.model

        print('adding version to project ...')
        model.add_version(
            os.path.join(config.save_location, 'best_miou_model.pt'),
            categories=project.categories,
            architecture="ml4vision-seg",
            params = {
                'min_size': config.transform.min_size if config.transform.resize else False,
                'pad': 32,
                'normalize': True,
            },
            metrics = {
                'miou': round(state['metrics']['mean_iou'], 3)
            }
        )
