"""
3DPECP-Net
=============================

**Author: Wang Kang  **
**Email: ichbin.wkang@gmail.com **
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image
import timm.models as tm
import argparse
# 衡量误差
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square


# set train and val data prefixs
# prefixs = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2']
prefixs = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D2']


# prefixs = ['A1','B1','C2', 'D2', 'E1']

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = []
            self.img_label = []
            # self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            # self.img_label = [float(line.strip().split('\t')[1]) for line in lines]
            for line in lines:
                img_name = line.strip().split('\t')[0]
                img_prefix = img_name.split('-')[0]
                # if img_prefix is not None:
                if img_prefix in prefixs:
                    self.img_name.append(os.path.join(img_path, img_name))
                    self.img_label.append(float(line.strip().split('\t')[1]))
        ln = len(self.img_label)
        y = self.img_label
        y = torch.tensor(y)
        y = y.numpy()
        print(np.shape(y))
        mean_val = np.mean(y)
        std_val = np.std(y)
        y = (y - mean_val) / std_val
        self.img_label = y
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label


def loadCol(infile, k):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        dataset.append(temp2[k])
    if dataset[0].split('.')[-1] != 'png':
        dataset = [float(s) for s in dataset]
    return dataset


def loadColStr(infile, k):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        if temp2[0].split('-')[0] in prefixs:
            dataset.append(float(temp2[k]))
    # for i in range(0, len(dataset)):
    #     for j in range(k):
    #         #dataset[i].append(float(dataset[i][j]))
    #         dataset[i][j] = float(dataset[i][j])
    return dataset


def Normalize(data):
    # res = []
    data = np.array(data)
    meanVal = np.mean(data)
    stdVal = np.std(data)
    res = (data - meanVal) / stdVal
    return res, meanVal, stdVal


def InvNormalize(data, meanVal, stdVal):
    # data = data.cpu().numpy()
    data = np.array(data)
    return (data * stdVal) + meanVal


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # mkdir output path
    out_path = './output'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    image_datasets = {x: customData(img_path=infile,
                                    txt_path=os.path.join(infile, (x + '.txt')),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

    # wrap your data and label into Tensor
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size,
                                                       shuffle=True if x == 'train' else False,
                                                       # num_workers=1
                                                       ) for x in ['train', 'val']}  # 这里的shuffle可以打乱数据顺序！！！


    # 绘制训练和验证的损失曲线
    plt.figure()
    plt.title("Train and val Loss history vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), train_hist, label="train_hist")
    plt.plot(range(1, num_epochs + 1), val_hist, label="val_hist")
    # plt.ylim((0, 2.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(os.path.join('./output/', (
                'Hist of ' + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(
            batch_size) + '.png')))
    plt.show()
    hist = np.vstack((train_hist, val_hist))
    np.savetxt("./output/Hist of " + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(
        batch_size), hist.T)
    # np.savetxt("./output/train_hist of " + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size), train_hist)
    # np.savetxt("./output/val_hist of " + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size), val_hist)

    #######################################################################
    # -----------------------Evaluation--------------------------------
    ### 逆归一化，输出 val 的‘预测的结果’ ###
    test_lab = loadColStr(os.path.join(infile, 'val.txt'), 1)
    _, meanVal, stdVal = Normalize(test_lab)
    result = InvNormalize(result, meanVal, stdVal)

    ### 逆归一化，输出 test 的‘预测的结果’ ###
    # test_img_path = loadCol(os.path.join(infile, 'test.txt'), 0)
    # test_lab = loadCol(os.path.join(infile, 'test.txt'), 1)
    # _, meanVal, stdVal = Normalize(test_lab)
    #
    # model_ft.eval()
    # torch.no_grad()
    # result = []
    # transform = transforms.Compose([
    #     transforms.Resize(input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # for i in range(len(test_img_path)):
    #     test_img = Image.open(os.path.join(infile, test_img_path[i])).convert('RGB')
    #     test_img = transform(test_img).unsqueeze(0)
    #     test_img = test_img.to(device)
    #     out = model_ft(test_img)
    #     out = out.detach().cpu().numpy()
    #     out_v = out[0][0] * stdVal + meanVal
    #     # print(out_v)
    #     result.append(out_v)

    # 绘制期望和预测的结果
    plt.figure()
    plt.title(model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + "Validation Result")
    ts = range(len(test_lab))
    plt.plot(ts, test_lab, label="test_lab")
    plt.plot(ts, result, label="pred_lab")
    # plt.xlim((0, 200))
    # plt.ylim((0, 450000))
    plt.legend()
    plt.show()
    # 输出预测的结果到txt文件
    res = np.vstack((test_lab, result))
    np.savetxt('./output/' + 'Results of ' + fn + "_" + model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(
        batch_size), res.T, fmt='%s')

    ### 分析'每层误差信息' ###
    result = np.array(result)
    test_lab = np.array(test_lab)
    print('[每层误差信息]')
    error = (result - test_lab) / test_lab
    print('Mean(error): {:.2%}.'.format(np.mean(error)))
    print('Max(error): {:.2%}.'.format(np.max(error)))
    print('Min(error): {:.2%}.'.format(np.min(error)))
    print('Std(error): {:.2f}.'.format(np.std(error)))
    # 调用误差 RMSE, Mae, R^2
    Rs = mean_squared_error(test_lab, result) ** 0.5
    Mae = mean_absolute_error(test_lab, result)
    R2_s = r2_score(test_lab, result)
    print('Root mean_squared_error: {:.2f}J, Mean_absolute_error: {:.2f}, R2_score: {:.2f}.'.format(Rs, Mae, R2_s))

    ### 分析'总误差信息' ###
    print('[总误差信息]')
    E1 = np.sum(test_lab)
    E2 = np.sum(result)
    Er = (E1 - E2) / E2
    print('Actual total EC: {:.2f}J, Predicted total EC: {:.2f}J, Er: {:.2%}'.format(E1, E2, Er))

    res_error = [np.mean(error), np.max(error), np.min(error), np.std(error), E1, E2, Er, Rs, Mae, R2_s]
    np.savetxt('./output/' + 'Error of ' + fn + "_" + model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(
        batch_size),
               np.array(res_error), fmt='%s')
