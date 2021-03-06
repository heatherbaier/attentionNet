import numpy as np
import random
import torch



import numpy as np
import random
import torch



class channelMaxPool(torch.nn.Module):
    def __init__(self, in_channels, h, w, batch_size):
        super(channelMaxPool, self).__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = h


    def forward(self, x):
        # torch.reshape(v, (1,2,2))
        x, i = torch.max(x, dim = 1)
        return torch.reshape(x, (x.shape[0], 1, self.h, self.w))


class spatialMaxPool(torch.nn.Module):
    def __init__(self, in_channels, batch_size):
        super(spatialMaxPool, self).__init__()
        self.in_channels = in_channels
        self.batch_size = batch_size

    def forward(self, x):
        x, i = torch.max(x, dim = -1)
        x, i = torch.max(x, dim = -1)
        return torch.reshape(x, (x.shape[0], self.in_channels, 1, 1))


class channelAvgPool(torch.nn.Module):
    def __init__(self, in_channels, h, w, batch_size):
        super(channelAvgPool, self).__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = h


    def forward(self, x):
        x = torch.mean(x, dim = 1)
        return torch.reshape(x, (x.shape[0], 1, self.h, self.w))


class spatialAvgPool(torch.nn.Module):
    def __init__(self, in_channels, batch_size):
        super(spatialAvgPool, self).__init__()
        self.in_channels = in_channels
        self.batch_size = batch_size

    def forward(self, x):
        x = torch.mean(x, dim = -1)
        x = torch.mean(x, dim = -1)
        return torch.reshape(x, (x.shape[0], self.in_channels, 1, 1))



class attnNet(torch.nn.Module):
    def __init__(self, in_channels, h, w, batch_size, resnet):
        super(attnNet, self).__init__()

        # Normal resnet stuff
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.linear = torch.nn.Linear(in_features=2048, out_features=1, bias = True)


        # Attention layers
        self.sMP = spatialMaxPool(in_channels = 2048, batch_size = batch_size)
        self.cMP = channelMaxPool(in_channels = 2048, h = 7, w = 7, batch_size = batch_size)
        self.sAP = spatialAvgPool(in_channels = 2048, batch_size = batch_size)
        self.cAP = channelAvgPool(in_channels = 2048, h = 7, w = 7, batch_size = batch_size)
        # self.out_channels = int(in_channels/16)
        self.out_channels = in_channels
        self.convR_M = torch.nn.Conv2d(in_channels = 2048, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convA_M = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convB_M = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (3,3), bias=True, padding = 1)
        self.convC_M = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (7,7), bias=True, padding = 3)
        self.convE_M = torch.nn.Conv2d(in_channels = self.out_channels * 3, out_channels = 2048, kernel_size = (1,1), bias=True)
        
        self.convR_A = torch.nn.Conv2d(in_channels = 2048, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convA_A = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convB_A = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (3,3), bias=True, padding = 1)
        self.convC_A = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (7,7), bias=True, padding = 3)
        self.convE_A = torch.nn.Conv2d(in_channels = self.out_channels * 3, out_channels = 2048, kernel_size = (1,1), bias=True)

        self.bn2 = torch.nn.BatchNorm2d(2048)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Max Pooling
        fsM = self.sMP(out)
        fcM = self.cMP(out)
        fscM = torch.mul(fsM, fcM)
        rM = self.convR_M(fscM)
        aM = self.convA_M(rM)
        bM = self.convB_M(rM)
        cM = self.convC_M(rM)
        catM = torch.cat((aM,bM,cM), dim = 1)
        eM = self.convE_M(catM)

        # Avg Pooling
        fsA = self.sAP(out)
        fcA = self.cAP(out)
        fscA = torch.mul(fsA, fcA)
        rA = self.convR_A(fscA)
        aA = self.convA_A(rA)
        bA = self.convB_A(rA)
        cA = self.convC_A(rA)
        catA = torch.cat((aA,bA,cA), dim = 1)
        eA = self.convE_A(catA)

        added = torch.add(eA, eM)
        added_norm = self.bn2(added)
        out = self.sigmoid(added_norm)

        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.linear(out)

        return out