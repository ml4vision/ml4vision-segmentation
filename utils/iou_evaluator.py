import torch

class IOUEvaluator:

    def __init__(self, n_classes, ignore_index=255):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset (self):
        self.tp = torch.zeros(self.n_classes).double()
        self.fp = torch.zeros(self.n_classes).double()
        self.fn = torch.zeros(self.n_classes).double()        

    # sizes should be "batch_size x 1 x H x W"
    def addBatch(self, x, y):   # x=preds, y=targets
        # clone as we will change x & y if ignore
        x = x.clone()
        y = y.clone()

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        # remove ignore label from target & pred by setting both target and pred to class 0
        ignore_mask = (y == self.ignore_index)
        ignores = ignore_mask.sum().item()
        x[ignore_mask] = 0
        y[ignore_mask] = 0

        # scatter pred "batch_size x 1 x H x W"  to onehot
        x_onehot = torch.zeros(
            x.size(0), self.n_classes, x.size(2), x.size(3))
        if x.is_cuda:
            x_onehot = x_onehot.cuda()
        x_onehot.scatter_(1, x, 1).float()

        # scatter target "batch_size x 1 x H x W"  to onehot
        y_onehot = torch.zeros(
            y.size(0), self.n_classes, y.size(2), y.size(3))
        if y.is_cuda:
            y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, y, 1).float()

        # times prediction and gt coincide is 1
        tpmult = x_onehot * y_onehot
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        tp[0] = tp[0] - ignores
        
        # times prediction says its that class and gt says its not
        fpmult = x_onehot * (1-y_onehot) 
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        
        # times prediction says its not that class and gt says it is
        fnmult = (1-x_onehot) * (y_onehot)
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou).item(), list(iou.numpy())     # returns "iou mean", "iou per class"

    def getRecall(self):
        num = self.tp
        den = self.tp + self.fn + 1e-15
        recall = num / den
        return torch.mean(recall).item(), list(recall.numpy())     # returns "recall mean", "recall per class"