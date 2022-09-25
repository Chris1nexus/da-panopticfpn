


import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, sigmoid_focal_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DiscriminatorP(nn.Module):
    def get_block(n_channels):
        return nn.Sequential(
                        nn.Conv2d(n_channels, 
                                n_channels//2, 
                                kernel_size = (1, 1), 
                                stride = 2, bias=False),
                        nn.BatchNorm2d(n_channels//2),
                        nn.ReLU(inplace = True),
                        nn.Dropout()
                        )
    def __init__(self, n_channels=256, p_scale=2):
        super(DiscriminatorP, self).__init__()

        self.conv_nn_block = nn.Sequential(
                *[DiscriminatorP.get_block(n_channels//2**exponent ) for exponent in range(p_scale-1)   ],
                nn.AdaptiveAvgPool2d((1, 1))
        )#.cuda()
        self.out_block = nn.Sequential(
                nn.Linear(n_channels//2**(p_scale-1), 128, bias = False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace = True),
                nn.Dropout(),
                nn.Linear(128, 1, bias= False)
        )#.cuda() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, target_domain = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.conv_nn_block(x) 
        x = torch.flatten(x,1)
        x = self.out_block(x)
        if target_domain:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss

    
class DiscriminatorRes2(nn.Module):

    def __init__(self):

        super(DiscriminatorRes2, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = (1, 1) ,bias = False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 1, kernel_size=(1, 1), bias = False)
        )#.cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, target_domain = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        if target_domain:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
        return loss    


class DiscriminatorRes3(nn.Module):

    def __init__(self):

        super(DiscriminatorRes3, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = (1, 1) ,bias = False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size = (1, 1) ,bias = False),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, kernel_size=(1, 1), bias = False)
        )#.cuda()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, target_domain = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        if target_domain:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t, alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s, alpha=0.25,gamma=2,reduction="mean")
        return loss


class DiscriminatorRes4(nn.Module):

    def __init__(self):

        super(DiscriminatorRes4, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(512, 128, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )#.cuda()
        self.reducer2 = nn.Linear(128, 1, bias = False ).cuda()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, target_domain = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        if target_domain:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss

    

class DiscriminatorRes5(nn.Module):

    def __init__(self):
        super(DiscriminatorRes5, self).__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(1024, 256, kernel_size = (3, 3), stride = 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 2, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )#.cuda()
        self.reducer2 = nn.Sequential(
            nn.Linear(256, 128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(128, 1, bias= False)
        )#.cuda() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, target_domain = False, alpha = 1):
        x = GradReverse.apply(x, alpha)
        x = self.reducer(x) 
        x = torch.flatten(x,1)
        x = self.reducer2(x)
        if target_domain:
            domain_t = torch.ones(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_t,alpha=0.25,gamma=2,reduction="mean")
        else:
            domain_s = torch.zeros(x.size()).float().cuda()
            loss = sigmoid_focal_loss_jit(x,domain_s,alpha=0.25,gamma=2,reduction="mean")
        return loss

