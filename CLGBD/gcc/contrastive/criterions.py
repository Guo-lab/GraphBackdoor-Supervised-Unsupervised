import torch
from torch import nn


class NCESoftmaxLoss(nn.Module): #@ MOCO
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x): # (32, 16385)
        #//print(x, x.shape)
        #//exit()
        bsz = x.shape[0]
        x = x.squeeze()
        #     print("in NCELoss: ", x, x.shape)
        #     in NCELoss:  
        #    tensor([[ 9.6569, -2.3864, -2.3645,  ...,  0.8471, -1.0484, -0.1975],
        #     [10.5762, -1.6900, -2.0508,  ...,  2.8406, -2.5752, -1.4642],
        #     [ 8.1413, -2.4482, -1.7724,  ..., -0.8273, -2.0916,  1.5445],
        #     ...,
        #     [ 9.1587, -3.3493, -4.4201,  ...,  3.5501, -4.3156,  0.5444],
        #     [11.3684, -3.5302, -2.1568,  ...,  0.8463,  0.4349, -0.8279],
        #     [ 8.1873, -1.3063, -0.9368,  ...,  0.5790, -0.2010, -1.3866]],
        #    grad_fn=<SqueezeBackward0>) torch.Size([32, 16385])
        
        # label = torch.zeros([bsz]).cuda().long()
        label = torch.zeros([bsz]).type(torch.LongTensor) ## cuda 的时候是这个样子的 label = torch.zeros([bsz]).cuda().long()
        #//print(label, label.shape)
        loss = self.criterion(x, label) # 这是因为，所有的 label 都要以第一个为正确判断的 label
        #//print(loss, loss.shape)
        return loss


class NCESoftmaxLossNS(nn.Module): #@ E2E
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        # label = torch.arange(bsz).cuda().long()
        label = torch.arange(bsz)## .cuda().long()
        loss = self.criterion(x, label)
        return loss
