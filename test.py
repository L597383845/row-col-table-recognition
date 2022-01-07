# -*- coding: utf-8 -*-
# cython: language_level=3
'''
@version : 0.1
@Author : Charles
@Time : 2020/3/12 下午4:09
@File : test.py
'''
import time
import torch
import cv2 as cv
import numpy as np
from Config import Config
from trans_model import  CNN_bigru,CNN_bilstm
from data_generator import TableDataset
from split_model import Split,split_loss
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
threshold_col = 0.5
threshold_row = 0.5
args = Config()
model_path = "./"+args.model_name+"/split_epoch_13.pth"

if args.model_name == "cnn_bilstm":
    mymodel = CNN_bilstm()
if args.model_name == "cnn_bigru":
    mymodel = CNN_bigru()
if args.model_name == "split":
    mymodel = Split()
if torch.cuda.is_available():
    mymodel = mymodel.cuda()
print('loading pretrained model from {0}'.format(model_path))
state_dict = torch.load(model_path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k[7:] # remove `module.`
    new_state_dict[namekey] = v
mymodel.load_state_dict(new_state_dict)


def table_reg(image, model):
    #image = image.transpose(2, 0, 1)  # h,w,c -> c,h,w
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image)
    image.sub_(args.mean).div_(args.std)

    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    model.eval()
    if args.model_name=="split":
        _,_,pred_row, _,_,pred_col = model(image)
        #print(pred_row)
        pred_row = pred_row.cpu().detach().numpy()[0]
        pred_col = pred_col.cpu().detach().numpy()[0]
    else:
        pred= model(image)
        pred = pred.cpu().detach().numpy()[0]
        pred = pred[:,0]

        #print(pred)
        pred_row, pred_col = pred[:600], pred[600:]
    #print(pred_row)
    #print(pred_col)
    #print(pred_col)
    #print(pred_row[:,0])
    #pred_col = pred_col.cpu().detach().numpy()[0]
    pred_col[pred_col >= threshold_col] = 1
    pred_col[pred_col < threshold_col] = 0

    pred_row[pred_row >= threshold_row] = 1
    pred_row[pred_row < threshold_row] = 0
    row_line_index = np.nonzero(pred_row)[0]
    col_line_index = np.nonzero(pred_col)[0]
    #print(row_line_index)
    #print(col_line_index)
    return row_line_index,col_line_index


def center_line(line_index):
    # 通过表格线区域找出此区域的中点
    res = []
    if len(line_index) == 0:
        return res
    tmp_index = [line_index[0]]
    for i in range(1, len(line_index)):

        if line_index[i] <= line_index[i-1] + 5:
            tmp_index.append(line_index[i])
        else:
            res.append(int(np.median(tmp_index)))
            tmp_index = [line_index[i]]
    if tmp_index:
        res.append(int(np.median(tmp_index)))
    return res

def draw_line(image, row_line_index1, col_line_index1, save_path):
    h, w, c = image.shape
    # 找出中点
    #cv.imwrite(save_path.replace(".png","1.png"), image)
    row_line_index = center_line(row_line_index1)
    col_line_index = center_line(col_line_index1)
    # 解决两条线离得很近的情况（实际上有一条不是表格线）
    # todo
    for i in row_line_index:
        cv.line(image, (0, i), (w, i), color=(0, 0, 255), thickness=2)
    for j in col_line_index:
        cv.line(image, (j, 0), (j, h), color=(0, 255, 0), thickness=2)

    f_dict={
        "row_label_pr":row_line_index1.tolist(),
        "col_label_pr":col_line_index1.tolist(),
        "row_pr": row_line_index,
        "col_pr": col_line_index
    }
    #with open(save_path.replace(".jpg", ".json").replace(".png",".json").replace(".PNG",".json"), "w", encoding="utf-8") as f:
    #    json.dump(f_dict,f)
    # image = cv.resize(image, None, fx=1/ratio, fy=1/ratio, interpolation=cv.INTER_CUBIC)
    print(save_path)
    cv.imwrite(save_path, image)





def image_resize(image,size, new_size):
    image=cv.resize(image,(600,600),cv.INTER_AREA)
    return image


def get_re(image_path, save_path):
    image = cv.imread(image_path)
    h, w, c = image.shape
    image=image_resize(image,(h,w),(600,600))
    # new_h, new_w = 600, 600     # 限制高和宽最大为 600
    # new_w, new_h, ratio = image_resize((h, w), (new_h, new_w))
    # image_split = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    # row_line_index, col_line_index = table_reg(image_split, split)
    row_line_index, col_line_index = table_reg(image, mymodel)
    # draw_line(image_split, row_line_index, col_line_index, save_path, ratio)
    draw_line(image, row_line_index, col_line_index, save_path)
    end_time = time.time()
    print('Use Time : {}'.format(end_time - begin_time))


if __name__ == '__main__':
    begin_time = time.time()
    file_path = "./tests/"
    o_path = "./results/"
    files = os.listdir(file_path)

    print(len(files))
    for file in files:
        get_re(file_path + file, o_path + file)


'''
if __name__ == '__main__':
    begin_time = time.time()
    dataset_name = "icdar19"

    if dataset_name == "unlv":
        file_path = "/data/unlv/"
    if dataset_name == "icdar13":
        file_path = "/data/icdar13/"
    if dataset_name == "icdar19":
        file_path = "/data/icdar19_table/"
    if dataset_name == "pubtabnet_train":
        file_path = "/data/PubTabNetV2/train/"
    if dataset_name =="pubtabnet_val":
        file_path = "/data/PubTabNetV2/val/"
    if dataset_name =="pubtabnet_minival":
        file_path = "/data/PubTabNetV2/mini_val/"
    if dataset_name =="pubtabnet_test":
        file_path = "/data/PubTabNetV2/test/"
    if dataset_name =="pubtabnet_final":
        file_path = "/data/PubTabNetV2/final_eval/"

    if dataset_name == "pubtabnet_complex":
        with open("/home/lyb/TableCompetition/TENER/SimpleComplex.json","r",encoding="utf8") as f:
            all_data = json.load(f)



    o_path = "/data/rc_result/" + dataset_name + "/" + args.model_name + "/"
    if not os.path.exists(o_path):
        os.mkdir(o_path)

    if dataset_name=="pubtabnet_complex":
        files = all_data.keys()
    else:
        files = os.listdir(file_path)

    print(len(files))
    for file in files:
        if dataset_name=="pubtabnet_complex":
            if all_data[file]['split']=='train':
                file_path = '/data/PubTabNetV2/train/'
                if all_data[file]['type']=='complex':
                    get_re(file_path + file, o_path + file)
            continue


        if dataset_name =="pubtabnet":
            if file[-4:] =="json":
                file_path = '/data/PubTabNetV2/train/'
                get_re(file_path+file.replace('json','png'),o_path+file.replace('json','png'))
        else:
            #if file=="PMC5207245_008_00.png":
            #    get_re(file_path + file, o_path + file)
            #else:
            #    continue
            if file[-3:] == "png" or file[-3:] == "jpg":
                get_re(file_path + file, o_path + file)

'''
