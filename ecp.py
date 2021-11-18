""" evalEC
Copyright 2021 Wang Kang
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

from torch.utils.data import Dataset
from PIL import Image

import timm as tm
from timm.models import create_model
from timm.utils import *

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import pandas as pd


# set train and val data prefixs
prefixs = None
# prefixs = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2']
# prefixs = ['A1','A2','B1','B2','C1','C2', 'D2', 'E1']
# prefixs = ['A1','B1','C2', 'D2', 'E1']


def initialize_model(model_name, num_classes=1, feature_extract=False, use_pretrained=False, drop=0, drop_path=0.1, drop_block=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # if model_name == "resnet":
    #     """ Resnet34
    #     """
    #     model_ft = tm.resnet34(pretrained=use_pretrained)
    #     set_parameter_requires_grad(model_ft, feature_extract)
    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #     input_size = 224

    elif model_name == "regnet":
        """ regnet
            regnety_040， regnety_080， regnety_160
        """
        model_ft = tm.regnety_040(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "efficientnet_b2":
        """ 
            efficientnet_b2 256, efficientnet_b3 288, efficientnet_b4 320
        """
        model_ft = tm.efficientnet_b2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 256

    elif model_name == "efficientnet_b3":
        """ 
            efficientnet_b2 256, efficientnet_b3 288, efficientnet_b4 320
        """
        model_ft = tm.efficientnet_b3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 288

    elif model_name == "efficientnet_b4":
        """ 
            efficientnet_b2 256, efficientnet_b3 288, efficientnet_b4 320
        """
        model_ft = tm.efficientnet_b4(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 320

    elif model_name == "vit":
        """ vit
        """
        model_ft = tm.vit_tiny_patch16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "pit":
        """ pit
        pit_xs_224, pit_s_224
        """
        model_ft = tm.pit_xs_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.get_classifier().in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "deit":
        """ deit
            deit_small_patch16_224, deit_base_patch16_224
        """
        model_ft = tm.deit_small_patch16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mixer":
        """ mixer
        """
        model_ft = tm.mixer_b16_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "swin-vit":
        """ swin-vit
        tm.swin_tiny_patch4_window7_224, tm.swin_small_patch4_window7_224
        """
        model_ft = tm.swin_small_patch4_window7_224(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "conformer":
        """
        Conformer_tiny_patch16, Conformer_small_patch16, Conformer_small_patch32, Conformer_base_patch16
        """
        model_ft = create_model(
            "Conformer_base_patch16",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        for param in model_ft.conv_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.trans_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        input_size = 224


    elif model_name == "ecpnet":
        model_ft = create_model(
            "EcpNet_tiny_patch16",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        for param in model_ft.conv_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.trans_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.mlp_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        input_size = 224

    elif model_name == "ecpnetno":
        model_ft = create_model(
            "EcpNet_NoConnect",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        for param in model_ft.conv_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.trans_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.mlp_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        input_size = 224

    elif model_name == "ecpnetlv":
        model_ft = create_model(
            "EcpNet_LCIvec",
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        for param in model_ft.conv_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.trans_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.mlp_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        input_size = 224


    elif model_name == "ecpnetv":
        model_ft = create_model(
            "EcpNet_Vec",
            pretrained=use_pretrained,
            num_classes=num_classes,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        for param in model_ft.mlp_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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
                # img_prefix = img_name.split('-')[0]
                # if (img_prefix is not None) and (img_prefix not in prefixs):
                #     continue
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
        # if temp2[0].split('-')[0] in prefixs:
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

    '''
        training each epoch
    '''
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
    plt.title(model_name + "_" + "Validation Result")
    ts = range(len(test_lab))
    plt.plot(ts, test_lab, label="test_lab")
    plt.plot(ts, result, label="pred_lab")
    plt.legend()
    plt.show()
    res = np.vstack((test_lab, result))
    np.savetxt(os.path.join(save_path, 'Results' + "_" + str(batch_size)), res.T, fmt='%s')

    '''layered error'''
    result = np.array(result)
    test_lab = np.array(test_lab)
    # writer.add_scalars('Validation/result', {'test_lab': test_lab, 'pred_lab': result}, ts)
    print('[Layered error]')
    error = (result - test_lab) / test_lab
    print('Mean(error): {:.2%} | Max(error): {:.2%} | Min(error): {:.2%} | Std(error): {:.2f}'.format(
        np.mean(error), np.max(error), np.min(error), np.std(error)))

    '''RMSE, Mae, R^2'''
    print('[Statistic error]')
    Rs = mean_squared_error(test_lab, result) ** 0.5
    Mae = mean_absolute_error(test_lab, result)
    R2_s = r2_score(test_lab, result)
    print('Root mean_squared_error: {:.2f}J | Mean_absolute_error: {:.2f} | R2_score: {:.2f}.'.format(Rs, Mae, R2_s))

    '''total error'''
    print('[Total error]')
    E1 = np.sum(test_lab)
    E2 = np.sum(result)
    Er = np.abs((E1 - E2) / E2)
    print('Actual EC: {:.2f}J | Predicted EC: {:.2f}J | Er: {:.2%}'.format(E1, E2, Er))

    '''save error to file'''
    res_error = [np.mean(error), np.max(error), np.min(error), np.std(error), E1, E2, Er, Rs, Mae, R2_s]
    np.savetxt(os.path.join(save_path, 'Error' + "_" + str(batch_size)), np.array(res_error), fmt='%s')


'''write to excel'''
def save_excel(data, file):
    writer = pd.ExcelWriter(file)  # 写入Excel文件
    data = pd.DataFrame(data)
    data.to_excel(writer, sheet_name='Sheet1', float_format='%.2f', header=False, index=False)
    writer.save()
    writer.close()


