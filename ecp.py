""" evalEC
Copyright 2021 Wang Kang
"""

from __future__ import print_function
from __future__ import division
import torch

import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

from torch.utils.data import Dataset
from PIL import Image


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



# set train and val data prefixs
# prefixs = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2']
prefixs = ['A1','A2','B1','B2','C1','C2', 'D2', 'E1']
# prefixs = ['A1','B1','C2', 'D2', 'E1']


# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = []
            self.img_label = []
            self.params = []
            for line in lines:
                ls = line.strip().split('\t')
                img_name = ls[0]
                img_prefix = img_name.split('-')[0]
                # if img_prefix is not None:
                if img_prefix in prefixs:
                    self.img_name.append(os.path.join(img_path, img_name))
                    self.img_label.append(float(ls[1]))
                    self.params.append([float(ls[2]), float(ls[3]), float(ls[4]), float(ls[5])])
                    # self.params.append([float(ls[2]), float(ls[3])])
        y = self.img_label
        y,_,_ = Normalize(y)
        print('[' + dataset+ ']')
        print('img_label shape: {}'.format(np.shape(y)))

        z = self.params
        z,_,_ = Normalize(z)
        print('params shape: {}'.format(np.shape(z)))
        self.img_label = y
        self.params = z
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        p = self.params[item]
        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label, p


def loadColStr(infile, k):
    f = open(infile, 'r')
    sourceInLine = f.readlines()
    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        if temp2[0].split('-')[0] in prefixs:
            dataset.append(float(temp2[k]))
    return dataset


def Normalize(data):
    # res = []
    data = torch.tensor(data)
    data= data.numpy()
    res = data
    aVal = 0
    bVal = 0
    # [1] mean-std norm
    if len(np.shape(data)) == 1:
        aVal = np.mean(data)
        bVal = np.std(data)
        res = (data - aVal) / bVal
    elif len(np.shape(data)) == 2:
        if data[0, 0] > 1:
            for i in [0, 1]:
                aVal = np.mean(data[:, i])
                bVal = np.std(data[:, i])
                res[:, i] = (data[:, i] - aVal) / bVal
    else:
        for i in [2, 3]:
            aVal = np.mean(data[:, i])
            bVal = np.std(data[:, i])
            res[:, i] = (data[:, i] - aVal) / bVal
    # [2] 0-1 norm
    # if len(np.shape(data)) == 1:
    #     aVal = np.min(data)
    #     bVal = np.max(data)
    #     res = (data-aVal)/(bVal-aVal)
    # else:
    #     for i in [2,3]:
    #         aVal = np.min(data[:,i])
    #         bVal = np.max(data[:,i])
    #         res[:, i] = (data[:, i]-aVal)/(bVal-aVal)
    # bVal = bVal-aVal
    # [3] max norm
    # if len(np.shape(data)) == 1:
    #     bVal = np.max(data)
    #     res = data / bVal
    # else:
    #     for i in [2, 3]:
    #         bVal = np.max(data[:, i])
    #         res[:, i] = data[:, i] / bVal
    # aVal = 0
    return res, aVal, bVal


def InvNormalize(data, aVal, bVal):
    # data = data.cpu().numpy()
    data = np.array(data)
    return (data*bVal)+aVal


def eval_EC(model_name, model_ft, save_path, infile, phase, batch_size=16, input_size=224):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
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
        'test': transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")
    image_datasets = customData(img_path=infile,
                                txt_path=os.path.join(infile, phase + '.txt'),
                                data_transforms=data_transforms,
                                dataset=phase)

    # wrap your data and label into Tensor
    dataloaders_dict = torch.utils.data.DataLoader(image_datasets,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   )
    since = time.time()
    result = []
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft.eval()  # Set model to evaluate mode
    # For one epoch
    for epoch in range(1):
        # For one batch.
        for inputs, labels, paras in dataloaders_dict:
            inputs = inputs.to(device)
            labels = labels.to(device)
            paras = paras.to(device)
            # forward
            if 'ecpnet' in model_name:
                outputs = model_ft(inputs, paras)
                # outputs = model_ft(paras)
            else:
                outputs = model_ft(inputs)

            # output predicted results
            if isinstance(outputs, list):
                # Conformer or ...
                res = outputs[0]
                for i in range(1, len(outputs)):
                    res = res + outputs[i]
                res = res / len(outputs)

                temp = res.detach().cpu().numpy()
                for i in range(temp.shape[0]):
                    result.append(temp[i])
            else:
                temp = outputs.detach().cpu().numpy()
                for i in range(temp.shape[0]):
                    result.append(temp[i])
    time_elapsed = time.time() - since
    print()
    print('Test complete in time of {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # -----------------------Test result--------------------------------#
    test_lab = loadColStr(os.path.join(infile, phase + '.txt'), 1)
    _, aVal, bVal = Normalize(test_lab)
    result = InvNormalize(result, aVal, bVal)
    result = np.squeeze(result)  # squeeze dimension

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ####################################
    plt.figure()
    plt.title(model_name  + "_" + str(batch_size) + "Test Result")
    ts = range(len(test_lab))
    plt.plot(ts, test_lab, label="test_lab")
    plt.plot(ts, result, label="pred_lab")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Best test result of ' + "_" + model_name + '.png'))
    plt.show()
    res = np.vstack((test_lab, result))
    np.savetxt(os.path.join(save_path, 'Best test result of ' + "_" + model_name), res.T, fmt='%s')
    ####################################
    result = np.array(result)
    test_lab = np.array(test_lab)
    print('[Test layered error]')
    error = (result - test_lab) / test_lab
    print('Mean(error): {:.2%}.'.format(np.mean(error)))
    print('Max(error): {:.2%}.'.format(np.max(error)))
    print('Min(error): {:.2%}.'.format(np.min(error)))
    print('Std(error): {:.2f}.'.format(np.std(error)))
    Rs = mean_squared_error(test_lab, result) ** 0.5
    Mae = mean_absolute_error(test_lab, result)
    R2_s = r2_score(test_lab, result)
    print('Root mean_squared_error: {:.2f}J, Mean_absolute_error: {:.2f}, R2_score: {:.2f}.'.format(Rs, Mae, R2_s))
    print('[Test total models error]')
    E1 = np.sum(test_lab)
    E2 = np.sum(result)
    Er = (E1 - E2) / E2
    print('Actual total EC: {:.2f}J, Predicted total EC: {:.2f}J, Er: {:.2%}'.format(E1, E2, Er))
    res_error = [np.mean(error), np.max(error), np.min(error), np.std(error), Rs, Mae, R2_s, E1, E2, Er]
    np.savetxt(os.path.join(save_path, 'Test error of ' + "_" + model_name),
               np.array(res_error), fmt='%s')


