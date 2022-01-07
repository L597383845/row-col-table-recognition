import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Config import Config
from trans_model import CNN_bigru,CNN_bilstm
from data_generator import TableDataset
from split_model import Split,split_loss

#torch.backends.cudnn.enabled = False
import cv2 as cv
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
writer = SummaryWriter('./scalar')


def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
'''
def adjust_learning_rate(optimizer, epoch,lr):
    """设置学习率衰减 """
    lr = lr * (0.33 ** (epoch // 1))
    print("lr now is "+str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''
def adjust_learning_rate(optimizer):
    """设置学习率衰减 """
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print("lr now is "+str(lr))
    lr = lr*0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(mymodel,train_loader,criterion,optimizer,epoch,args):
    for p in mymodel.parameters():
        p.requires_grad = True
    mymodel.train()
    data_len = len(train_loader)
    min_loss = 1000
    min_loss_index = 0

    for i_batch, (image, label) in enumerate(train_loader):
        #new_image=(image*params.std+params.mean)*255
        #new_image=new_image.astype(np.int)
        #cv.imwrite("aa.jpg",new_image)
        #print()

        if args.model_name == "split":
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
            cost = split_loss(image,label,mymodel,criterion)
        else:
            if args.cuda:
                image = image.cuda()
                label = label.cuda()
            plabel = mymodel(image)
            #print(plabel.size())
            #print(label.size())
            plabel=plabel.type(torch.float)
            label=label.type(torch.float)
            cost = criterion(plabel[:,:,:],label[:,:,:])

        mymodel.zero_grad()
        cost.backward()
        optimizer.step()
        writer.add_scalar('train', cost, epoch* data_len + i_batch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * data_len + i_batch)
        if (i_batch + 1) % args.displayInterval == 0:
            print("[{}/{}][{}/{}] Loss: {}".format(epoch, args.niter, i_batch, data_len, cost))

            if cost > min_loss:
                if i_batch+epoch*data_len - min_loss_index > args.require_improvement:
                    min_loss_index = i_batch+epoch*data_len
                    adjust_learning_rate(optimizer)
                    torch.save(mymodel.state_dict(), '{}/split_{}.pth'.format(args.model_name, str(i_batch+epoch*data_len)+"_"+str(min_loss)))
            else:
                min_loss = cost
                min_loss_index = i_batch+epoch*data_len
                if min_loss_index > 500:
                    torch.save(mymodel.state_dict(),'{}/split_{}.pth'.format(args.model_name, str(i_batch+epoch*data_len) + "_" + str(min_loss)))




def main(mymodel,train_loader,criterion,optimier,n_epochs,args):
    epoch = 0
    print("begin main")
    while epoch < n_epochs:
        train(mymodel,train_loader,criterion,optimizer,epoch,args)
        #adjust_learning_rate(optimizer)
        #adjust_learning_rate(optimizer,epoch,args.learning_rate)
        #if epoch % args.saveModel == 0:
        #torch.save(mymodel.state_dict(), '{}/split_{}.pth'.format(args.model_name, epoch))
        torch.save(mymodel.state_dict(),
                   '{}/split_epoch_{}.pth'.format(args.model_name, str(epoch)))
        epoch+=1



if __name__=="__main__":
    args = Config()
    image_dir = args.images_dir
    json_dir = args.json_dir
    dataset =TableDataset(image_dir,json_dir,args)
    train_loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    criterion =torch.nn.BCELoss()
    if args.model_name=="cnn_bilstm":
        mymodel = CNN_bilstm()
    if args.model_name=="cnn_bigru":
        mymodel = CNN_bigru()
    if args.model_name=="split":
        mymodel = Split()


    if args.cuda:
        mymodel.cuda()
        mymodel = nn.DataParallel(mymodel,device_ids=args.device_ids)
        criterion = criterion.cuda()
    mymodel.apply(weights_init)

    if args.trained_model:
        print("loading pretrained model from {}".format(args.trained_model))
        state_dict = torch.load(args.trained_model)
        #from collections import OrderedDict
        #new_state_dict = OrderedDict()
        #for k, v in state_dict.items():
        #    namekey = k[7:]  # remove `module.`
        #    new_state_dict[namekey] = v
        #mymodel.load_state_dict(new_state_dict)
        mymodel.load_state_dict(state_dict)
    #optimizer = optim.SGD(mymodel.parameters(),lr=args.learning_rate, momentum=0.9)

    optimizer = optim.Adam(mymodel.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.98))
    main(mymodel,train_loader, criterion, optimizer,args.niter,args)

