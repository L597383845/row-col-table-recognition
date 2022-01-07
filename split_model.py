# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/9 上午11:11
@File : split.py
'''
import torch
import torch.nn as nn
import numpy as np


def split_loss(input, label_split,split,criterion):
    # criterion 是 nn.BCELoss()
    label_split = label_split.type(torch.float)
    label_r, label_c = label_split[:,:600,0],label_split[:,600:,0]
    #print(label_c.size())
    #print(label_r.size())

    # 分割模型
    rp3, rp4, rp5, cp3, cp4, cp5 = split(input)     # rp3 b*H  cp3 b*W
    #print(rp3.size())
    # 合并模型
    # merge_input, grid_struct = get_merge_input(input, rp5, cp5)
    # out_up2, out_up3, out_down2, out_down3, out_left2, out_left3, out_right2, out_right3, split_zone = merge(merge_input, grid_struct)
    # D2, D3, R2, R3 = calc_prob_matrix(out_up2, out_up3, out_down2, out_down3, out_left2, out_left3, out_right2, out_right3, split_zone)
    # 分割损失

    L_split_tot = criterion(rp5, label_r) + 0.25 * criterion(rp4, label_r) + 0.1 * criterion(rp3, label_r) + \
                  criterion(cp5, label_c) + 0.25 * criterion(cp4, label_c) + 0.1 * criterion(cp3, label_c)
    # 合并损失
    # L_merge_tot = criterion(D3, label_D) + 0.25 * criterion(D2, label_D) + \
    #               criterion(R3, label_R) + criterion(R2, label_R)
    # 总损失
    # L_tot = L_split_tot + L_merge_tot
    L_tot = L_split_tot
    return L_tot


def get_merge_input(input, row, col):
    B, C, H, W = input.size()
    rb, rh = row.size()
    cb, ch = col.size()
    assert rb == H and cb == W, "输入图像大小与Split模型输出的行列大小不一致"
    row_ex = row.reshape((1, 1, -1, 1)).expand(1, 1, -1, W)       # 拓展行概率[r] -> [r, r, ..., r]        b*h -> b*c*h*w   b=c=1
    col_ex = col.reshape((1, 1, 1, -1)).expand(1, 1, H, -1)       # 拓展列概率[c] -> [[c], [c], ..., [c]]  b*w -> b*c*h*w
    row_region = torch.zeros((H, W), dtype=torch.float32)
    col_region = torch.zeros((H, W), dtype=torch.float32)


def calc_prob_matrix(out_up2, out_up3, out_down2, out_down3, out_left2, out_left3, out_right2, out_right3, split_zone):
    # b, c, h, w = out_up2.size()
    # b*c*h*w  b=c=1
    out_h, out_w = len(split_zone[0])-1, len(split_zone[1])-1
    # 生成 MxN 的矩阵 u,d,l,r
    def grid_mean(input):
        out = torch.zeros((out_h, out_w), dtype=torch.float32)
        for i in range(out_h):
            row = (split_zone[0][i], split_zone[0][i + 1])
            for j in range(out_w):
                col = (split_zone[1][j], split_zone[1][j + 1])
                grid_mean_v = torch.mean(input[0, 0, row[0]:row[1], col[0]:col[1]])
                out[i, j] = grid_mean_v
        return out
    u2, u3, d2, d3 = grid_mean(out_up2), grid_mean(out_up3), grid_mean(out_down2), grid_mean(out_down3)
    l2, l3, r2, r3 = grid_mean(out_left2), grid_mean(out_left3), grid_mean(out_right2), grid_mean(out_right3)
    # 计算上下合并的概率
    D2 = u2[1:, :] * d2[:-1, :] / 2 + (u2[1:, :] + d2[:-1, :]) / 4
    D3 = u3[1:, :] * d3[:-1, :] / 2 + (u3[1:, :] + d3[:-1, :]) / 4
    # 计算左右合并的概率
    R2 = l2[:, 1:] * r2[:, :-1] / 2 + (l2[:, 1:] + r2[:, :-1]) / 4
    R3 = l3[:, 1:] * r3[:, :-1] / 2 + (l3[:, 1:] + r3[:, :-1]) / 4
    return D2, D3, R2, R3

def projection_pooling_row(input):
    b, c, h, w = input.size()
    ave_v = input.mean(dim=3)
    ave_v = ave_v.reshape(b, c, h, -1)
    input[:, :, :, :] = ave_v[:, :, :]
    return input


def projection_pooling_column(input):
    b, c, h, w = input.size()
    input = input.permute(0, 1, 3, 2)
    ave_v = input.mean(dim=3)
    ave_v = ave_v.reshape(b, c, w, -1)
    input[:, :, :, :] = ave_v[:, :, :]
    input = input.permute(0, 1, 3, 2)
    return input


class Block(nn.Module):
    def __init__(self, in_channels, i, row_column=0):
        super(Block, self).__init__()
        self.index = i
        self.row_column = row_column
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, padding=4, dilation=4)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        self.branch1 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels=18, out_channels=1, kernel_size=1)

    def forward(self, input):
        out1 = torch.cat([self.conv1(input), self.conv2(input), self.conv3(input)], dim=1)
        if self.index <= 3:
            if self.row_column == 0:
                out1 = self.pool1(out1)
            else:
                out1 = self.pool2(out1)
        if self.row_column == 0:
            b1 = projection_pooling_row(self.branch1(out1))     # 上分支的投影池化
            b2 = projection_pooling_row(self.branch2(out1))     # 下分支的投影池化
        else:
            b1 = projection_pooling_column(self.branch1(out1))  # 上分支的投影池化
            b2 = projection_pooling_column(self.branch2(out1))  # 下分支的投影池化
        b, c, h, w = b2.size()
        # b2 = b2.squeeze(1)
        b2 = torch.sigmoid(b2)
        output = torch.cat([b1, out1, b2], dim=1)
        return output, b2


class SFCN(nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        cnn = nn.Sequential()
        input_c = [3, 18, 18]
        padding = [3, 3, 6]
        dilation = [1, 1, 2]
        for i in range(3):
            cnn.add_module('sfcn{}'.format(i), nn.Conv2d(input_c[i], 18, 7, padding=padding[i], dilation=dilation[i]))
            cnn.add_module('sfcn_relu{}'.format(i), nn.ReLU(True))
        self.cnn = cnn

    def forward(self, input):
        output = self.cnn(input)
        return output


class Split(nn.Module):

    def __init__(self):
        super(Split, self).__init__()
        self.sfcn = SFCN()
        self.rpn()
        self.cpn()

    def rpn(self):
        self.row_1 = Block(18, 1)
        self.row_2 = Block(37, 2)
        self.row_3 = Block(37, 3)
        self.row_4 = Block(37, 4)
        self.row_5 = Block(37, 5)

    def cpn(self):
        self.column_1 = Block(18, 1, row_column=1)
        self.column_2 = Block(37, 2, row_column=1)
        self.column_3 = Block(37, 3, row_column=1)
        self.column_4 = Block(37, 4, row_column=1)
        self.column_5 = Block(37, 5, row_column=1)

    def forward(self, input):
        #print(input.shape)
        input = input.permute((0,3,1,2))
        out_fcn = self.sfcn(input)
        r1, rp1 = self.row_1(out_fcn)
        r2, rp2 = self.row_2(r1)
        r3, rp3 = self.row_3(r2)
        r4, rp4 = self.row_4(r3)
        r5, rp5 = self.row_5(r4)
        #print(r5.size(),rp5.size())

        c1, cp1 = self.column_1(out_fcn)
        c2, cp2 = self.column_2(c1)
        c3, cp3 = self.column_3(c2)
        c4, cp4 = self.column_4(c3)
        c5, cp5 = self.column_5(c4)
        # print(cp5[0, :, 0, :].size())
        return rp3[:, 0, :, 0], rp4[:, 0, :, 0], rp5[:, 0, :, 0], cp3[:, 0, 0, :], cp4[:, 0, 0, :], cp5[:, 0, 0, :]

if __name__ == '__main__':
    a = np.random.randint(0, 255, size=(1, 3, 500, 500))
    a = a.astype(np.float32)
    input = torch.from_numpy(a)
    split = Split()
    split = split.cuda()
    input = input.cuda()
    split(input)