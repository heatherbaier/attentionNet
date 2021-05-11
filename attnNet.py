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
        return torch.reshape(x, (self.batch_size, 1, self.h, self.w))


class spatialMaxPool(torch.nn.Module):
    def __init__(self, in_channels, batch_size):
        super(spatialMaxPool, self).__init__()
        self.in_channels = in_channels
        self.batch_size = batch_size

    def forward(self, x):
        x, i = torch.max(x, dim = -1)
        x, i = torch.max(x, dim = -1)
        return torch.reshape(x, (self.batch_size, self.in_channels, 1, 1))


class channelAvgPool(torch.nn.Module):
    def __init__(self, in_channels, h, w, batch_size):
        super(channelAvgPool, self).__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = h


    def forward(self, x):
        x = torch.mean(x, dim = 1)
        return torch.reshape(x, (self.batch_size, 1, self.h, self.w))


class spatialAvgPool(torch.nn.Module):
    def __init__(self, in_channels, batch_size):
        super(spatialAvgPool, self).__init__()
        self.in_channels = in_channels
        self.batch_size = batch_size

    def forward(self, x):
        x, i = torch.mean(x, dim = -1)
        x, i = torch.mean(x, dim = -1)
        return torch.reshape(x, (self.batch_size, self.in_channels, 1, 1))



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
        self.convR = torch.nn.Conv2d(in_channels = 2048, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convA = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (1,1), bias=True)
        self.convB = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (3,3), bias=True, padding = 1)
        self.convC = torch.nn.Conv2d(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = (7,7), bias=True, padding = 3)
        self.convE = torch.nn.Conv2d(in_channels = self.out_channels * 3, out_channels = 2048, kernel_size = (1,1), bias=True)
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
        # out = self.avgpool(out)
        # out = out.flatten(start_dim=1)
        # out = self.linear(out)


        # Max Pooling
        fsM = self.sMP(out)
        fcM = self.cMP(out)
        fscM = torch.mul(fsM, fcM)
        rM = self.convR(fscM)
        aM = self.convA(rM)
        bM = self.convB(rM)
        cM = self.convC(rM)
        catM = torch.cat((aM,bM,cM), dim = 1)
        eM = self.convE(catM)


        # Avg Pooling
        fsA = self.sAP(out)
        fcA = self.cAP(out)
        fscA = torch.mul(fsA, fcA)
        rA = self.convR(fscA)
        aA = self.convA(rA)
        bA = self.convB(rA)
        cA = self.convC(rA)
        catA = torch.cat((aA,bA,cA), dim = 1)
        eA = self.convE(catA)

        out = self.sigmoid(eA + eM)

        # print("A: ", out.shape)

        out = self.avgpool(out)

        # print("B: ", out.shape)

        out = out.flatten(start_dim=1)
        out = self.linear(out)

        return out