import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'cnn_bilstm'                    #/"split"/"cnn_bigru"/"cnn_bilstm"/
        self.images_dir = "../cnn_bigru_col/data2/PubTabNetV2/train/"
        self.json_dir = '../cnn_bigru_col/annojson/train/'
        self.trained_model = "./cnn_bilstm/split_epoch_100.pth"                            #"cnn_transformer/split_1.pth"
        self.device_ids =[0,1]
        self.cuda = True
        self.crf = True


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 10000                               # 若超过2000batch效果还没提升，则提前结束训练
        self.num_classes = 2                         # 类别数

        self.mean = 0.588
        self.std = 0.193
        self.num_epochs = 20                                            # epoch数

        self.batch_size = 2                                        # mini-batch大小
        self.saveModel = 1
        self.displayInterval = 20                                       # 查看loss
        self.sequence_size = 600                                        # 每句话处理成的长度(短填长切)
        self.embed = 600                                                # 每行/每列的feature长度
        self.learning_rate = 0.001005                                      # 学习率
        self.dim_model = 600                                            # target feature size
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2
        self.niter = 20
        self.beta1= 0.9