import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class clampLPIPS(nn.modules.loss._Loss):
    def __init__(self, device, reduction = 'mean'):
        super(clampLPIPS, self).__init__(reduction=reduction)
        self.lpipsloss =LearnedPerceptualImagePatchSimilarity(net_type = 'vgg', reduction = reduction, normalize = True).to(device)


    def forward(self, x,y, divisor = 1):
        clampx = torch.clamp(x, min=0, max=1)
        clampy = torch.clamp(y, min=0, max=1)
        mixedloss = self.lpipsloss(clampx,clampy) / divisor ** 2
        return mixedloss