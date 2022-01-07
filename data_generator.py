import pickle
import json
import os
import cv2 as cv
import numpy as np
import torch
from Config import Config
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

def draw_line(image, row_line_index, col_line_index, save_path, ratio):
    h, w, c = image.shape
    # 找出中点
    #print(col_line_index)
    #print(row_line_index)
    #row_line_index = center_line(row_line_index)
    #col_line_index = center_line(col_line_index)
    # 解决两条线离得很近的情况（实际上有一条不是表格线）
    # todo
    #print(row_line_index)
    #print(col_line_index)
    for i in row_line_index:
        cv.line(image, (0, i), (w, i), color=(0, 0, 255), thickness=1)
    for j in col_line_index:
        cv.line(image, (j, 0), (j, h), color=(0, 255, 0), thickness=1)
    # image = cv.resize(image, None, fx=1/ratio, fy=1/ratio, interpolation=cv.INTER_CUBIC)
    cv.imwrite("draw.jpg", image)


def center_line(line_index):
    # 通过表格线区域找出此区域的中点
    res = []
    #print(line_index)
    if len(line_index)==0:
        return res
    tmp_index = [line_index[0]]
    for i in range(1, len(line_index)):
        if line_index[i]>10000:
        #if line_index[i] <= line_index[i-1] + 5:
        #if line_index[i] == line_index[i-1] + 1:
            tmp_index.append(line_index[i])
        else:
            res.append(int(np.median(tmp_index)))
            tmp_index = [line_index[i]]
    if tmp_index:
        res.append(int(np.median(tmp_index)))
    return res

class TableDataset(Dataset):
    def __init__(self, images_dir, json_dir,config):
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.labels = []
        self.config = config
        self._get_labels()

    def _get_labels(self):
        with open('/home/lyb/TableCompetition/TENER/icdar19.pickle', 'rb') as fr:
            self.labels = pickle.load(fr)
    @staticmethod
    def _get_label(json_path):

        with open(json_path, 'r', encoding='utf-8') as f:
            lr = json.load(f)
        #print(lr)
        row_list= np.array(lr['row_label'])
        col_list = np.array(lr["col_label"])

        tmp = np.zeros((600, 2), dtype=float)
        if len(row_list)!=0:
            tmp[row_list, 0] = 1
        tmp[:, 1] = np.ones((600)) - tmp[:, 0]
        row_label = tmp

        tmp = np.zeros((600, 2), dtype=float)
        if len(col_list)!=0:
            tmp[col_list, 0] = 1
        tmp[:, 1] = np.ones((600)) - tmp[:, 0]
        col_label = tmp

        #print(row_label[:,0])
        #print(col_label[:,0])

        return torch.cat((torch.from_numpy(row_label), torch.from_numpy(col_label)),dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path, label_path = self.labels[item]
        #print(image_path)
        image = cv.imread(image_path)
        #cv.imwrite("1.jpg", image)
        new_w, new_h = 600, 600
        label = self._get_label(label_path)
        image = cv.resize(image,(new_w,new_h),interpolation=cv.INTER_CUBIC)

        #print("label[1]")
        #print(label[1][:,0])
        #col_index = np.where(label[0][:,0]==1)[0]
        #row_index = np.where(label[1][:,0]==1)[0]
        #print(col_index)
        #draw_line(image,col_index,row_index,1,1)
        #cv.imwrite("4.jpg", image)
        #exit()
        #print(label[0])
        #print(label[1])
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        image.sub_(self.config.mean).div_(self.config.std)

        #print(image.shape)
        #image =Variable(image.cuda())
        #label =Variable(label.cuda())
        #print(image.shape)
        return image, label #idx #headeridx

    '''
    def __getitem__(self, item):
        image_path, label = self.labels[item]
        image = cv.imread(image_path)
        image = image.transpose(2, 0, 1)  # h,w,c -> c,h,w
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        image.sub_(params.mean).div_(params.std)
        return image, label
    '''
