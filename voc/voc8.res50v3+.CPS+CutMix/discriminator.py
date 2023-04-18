# -*- coding = utf-8 -*-
# @Time = 2022/5/10 19:56
# Author = Chen
# @File = discriminator.py
# -- coding: utf-8 --
from torch.autograd import Variable
import torch.nn as nn


class s4GAN_discriminator(nn.Module):

    def __init__(self, num_classes, dataset, ndf = 64):
        super(s4GAN_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes + 3, ndf, kernel_size = 4, stride = 2, padding = 1)  # 160 x 160
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size = 4, stride = 2, padding = 1)  # 80 x 80
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size = 4, stride = 2, padding = 1)  # 40 x 40
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size = 4, stride = 2, padding = 1)  # 20 x 20
        if dataset == 'pascal_voc' or dataset == 'pascal_context':
            self.avgpool = nn.AvgPool2d((20, 20))
        elif dataset == 'cityscapes':
            self.avgpool = nn.AvgPool2d((16, 32))
        self.fc = nn.Linear(ndf * 8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('discriminator input size:')
        '''bs,24,512,512'''
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        '''bs,64,256,256'''
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        # print(x.size())
        '''bs,512,32,32'''
        maps = self.avgpool(x)
        conv4_maps = maps
        # print(conv4_maps.size())
        '''bs,512,1,1'''
        out = maps.view(maps.size(0), -1)
        # print('after view:')
        # print(out.size())
        '''bs,512'''
        # print('after fc:')
        # print(self.fc(out).size())
        '''bs,1'''
        out = self.sigmoid(self.fc(out))
        # print('discriminator output1 size:')
        # print(out.size())
        '''bs,1'''
        # print('discriminator output2 size:')
        # print(conv4_maps.size())
        '''bs,512,1,1'''
        return out, conv4_maps
