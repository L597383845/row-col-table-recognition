import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from Config import Config
import os


class CNN_bigru(nn.Module):
    def __init__(self):
        super(CNN_bigru,self).__init__()
        self.encoder = CNN_Encoder()
        self.cnn_1 = nn.Conv2d(96, 1, 3, padding=1)
        self.cnn_2 = nn.Conv2d(96, 1, 3, padding=1)
        self.decoder1 = Bi_GRU()
        self.decoder2 = Bi_GRU()
    def forward(self,x):

        x = self.encoder(x)
        x1 = self.cnn_1(x)
        x2 = self.cnn_2(x)
        #print(x1.shape)
        x1 = x1.reshape(x1.size()[0], x1.size()[2], x1.size()[3])
        x2 = x2.reshape(x2.size()[0], x2.size()[2], x2.size()[3])
        x1 = x1.permute((0, 2, 1))
        x2 = x2.permute((0, 1, 2))
        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        #print(x1.size())
        return torch.cat((x2,x1),dim=1)

class CNN_bilstm(nn.Module):
    def __init__(self):
        super(CNN_bilstm,self).__init__()
        self.encoder = CNN_Encoder()
        self.cnn_1 = nn.Conv2d(96, 1, 3, padding=1)
        self.cnn_2 = nn.Conv2d(96, 1, 3, padding=1)
        self.decoder1 = Bi_LSTM()
        self.decoder2 = Bi_LSTM()
    def forward(self,x):

        x = self.encoder(x)
        x1 = self.cnn_1(x)
        x2 = self.cnn_2(x)
        #print(x1.shape)
        x1 = x1.reshape(x1.size()[0], x1.size()[2], x1.size()[3])
        x2 = x2.reshape(x2.size()[0], x2.size()[2], x2.size()[3])
        x1 = x1.permute((0, 2, 1))
        x2 = x2.permute((0, 1, 2))
        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        #print(x1.size())
        return torch.cat((x2,x1),dim=1)


class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder,self).__init__()
        self.conv_0 = nn.Conv2d(3, 32, 3, stride=1, padding=1, dilation=1)
        #self.bn_0 = nn.BatchNorm2d(32)
        self.conv_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1)
        self.conv_2 = nn.Conv2d(32, 32, 3, stride=1, padding=2, dilation=2)
        self.conv_4 = nn.Conv2d(32, 32, 3, stride=1, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(96)
        self.conv_final = nn.Conv2d(96, 1, 3, padding=1)

    def forward(self, x):
        #print(x.size())
        x = x.permute((0,3,1,2))
        #print(x.size())
        x = self.conv_0(x)
        #x = self.bn_0(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x4 = self.conv_4(x)
        x = torch.cat((x1, x2, x4), 1)
        x = self.bn(x)
        return x

class Bi_LSTM(nn.Module):
    def __init__(self,hidden_size=512,num_layers=3):
        super(Bi_LSTM,self).__init__()
        self.n_hidden = hidden_size
        self.layers=num_layers
        self.bilstm = nn.LSTM(input_size=600,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
        self.dense = nn.Linear(hidden_size*2,2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,x):
        if not hasattr(self, '_flattened'):
            self.bilstm.flatten_parameters()
        #batch_size, h,w,c =x.size()
        #x=x.view(batch_size,h,w)
        x ,h_n = self.bilstm(x)
        x = self.dense(x)
        re = self.softmax(x)
        return re



class Bi_GRU(nn.Module):
    def __init__(self,hidden_size=512,num_layers=3):
        super(Bi_GRU,self).__init__()
        self.n_hidden = hidden_size
        self.layers=num_layers
        self.bigru = nn.GRU(input_size=600,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bidirectional=True)
        self.dense = nn.Linear(hidden_size*2,2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,x):
        if not hasattr(self, '_flattened'):
            self.bigru.flatten_parameters()
        #batch_size, h,w,c =x.size()
        #x=x.view(batch_size,h,w)
        x ,h_n = self.bigru(x)
        x = self.dense(x)
        re = self.softmax(x)
        return re


if __name__ == "__main__":
    os.environ["CUDA_VISUAL_DEVICES"] = "9"
    myconfig = Config()








