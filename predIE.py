""" PredIE
Copyright 2021 Wang Kang
"""

from __future__ import print_function
from __future__ import division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device_id = [0, 1, 2, 3]

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image

import timm.models as tm
from timm.utils import *
import argparse
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# from torchstat import stat

from timm.models import create_model
import models
from ecp import *


# Number of classes in the dataset
num_classes = 1

lr = 0.001

parm = {}  # 初始化保存模块参数的parm字典

# Set argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='../dataset/lattice_ec',
                    help='path to dataset')
parser.add_argument('--eval_phase', default='test', type=str, metavar='NAME',
                    help='Phase of eval dataset (default: )')
'''
Setting model and training params, some can use parser to get value.
Models to choose from [resnet, regnet, efficientnet, vit, pit, mixer, deit, swin-vit
alexnet, vgg, squeezenet, densenet, inception, Conformer_tiny_patch16, ecpnet]
'''
parser.add_argument('--model', default='ecpnetno', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet"')
parser.add_argument('-b', '--batch-size', type=int, default=240, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: )')
parser.add_argument('--use-pretrained', action='store_true', default=False,
                    help='Flag to use fine tuneing(default: False)')
parser.add_argument('--feature-extract', action='store_true', default=False,
                    help='False to finetune the whole model. True to update the reshaped layer params(default: False)')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')


def train_model(model, dataloaders, criterion, optimizer, GT, aVal, bVal, num_epochs=25, is_inception=False):
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    last_result = []
    best_result = []

    RE_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    best_epoch = 0
    min_loss = 999999
    min_metric = 999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        result = []

        '''
            Each epoch has a training and validation phase
        '''
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over batchs  data.
            for inputs, labels, paras in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                paras = paras.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                '''
                Forward
                track history if only in train
                '''
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        outputs = outputs.view(-1)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        if 'ecpnet' in args.model:
                            outputs = model(inputs, paras)
                            # outputs = model(paras)
                        else:
                            outputs = model(inputs)
                        if isinstance(outputs, list):
                            # Conformer or ...
                            for i, o in enumerate(outputs):
                                outputs[i] = o.view(-1)
                            loss_list = [criterion(o, labels) / len(outputs) for o in outputs]
                            loss = sum(loss_list)
                        else:
                            outputs = outputs.view(-1)
                            loss = criterion(outputs, labels)

                        '''output predicted results'''
                        if phase == 'val':
                            # result.append(outputs)
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
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # save loss value
                running_loss += loss.item() * inputs.size(0)

            '''
            After training or val one epoch
            '''
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                train_acc_history.append(epoch_loss)
            if phase == 'val':
                if epoch == num_epochs-1:
                    last_result = result
                val_acc_history.append(epoch_loss)

                # get every epoch error
                result = InvNormalize(result, aVal, bVal)
                Rs = mean_squared_error(GT, result) ** 0.5
                Mae = mean_absolute_error(GT, result)
                R2_s = r2_score(GT, result)
                print('RMSE: {:.2f}J | MAE: {:.2f} | R2_s: {:.2f}.'.format(Rs, Mae, R2_s))
                E1 = np.sum(GT)
                E2 = np.sum(result)
                Er = np.abs((E1 - E2) / E1)
                print('GT: {:.2f}J | ECP: {:.2f}J | Er: {:.2%}'.format(E1, E2, Er))
                RE_history.append(Er)

                # find best model
                # if epoch_loss < min_loss:
                #     min_loss = epoch_loss  # update min_loss
                #     print('Min loss: {:4f} in Epoch {}/{}'.format(min_loss, epoch+1, num_epochs))
                #     best_model_wts = copy.deepcopy(model.state_dict())
                #     best_epoch = epoch
                #
                #     # load best model weights, and return
                #     model.load_state_dict(best_model_wts)
                if Er < min_metric:
                    min_metric = Er  # update min_loss
                    print('Min Er: {:4f} in Epoch {}/{}'.format(Er, epoch+1, num_epochs))
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    best_result = result
        print()

    # After training or val all epochs
    time_elapsed = time.time() - since
    print('Training complete in time of {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights, and return
    model.load_state_dict(best_model_wts)

    # Save best model weights
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, 'Best_checkpoint.pth')
    torch.save(best_model_wts, save_path)

    return model, train_acc_history, val_acc_history, best_result, RE_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False





if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    # random_seed(666)

    '''get all args params'''
    args = parser.parse_args()
    infile = args.data_dir
    model_name = args.model
    batch_size = args.batch_size
    num_epochs = args.epochs

    '''define output path'''
    # t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    fn = args.data_dir.split('/')[-1]
    out_path = os.path.join('./outputs', fn, args.model)

    test_path = os.path.join(out_path, args.eval_phase)

    '''Initialize the model for this run'''
    model_ft, input_size = initialize_model(model_name, num_classes, args.feature_extract, args.use_pretrained,
                                            args.drop, args.drop_path, args.drop_block)

    '''
        Analyze flops and params
    '''
    if args.model in ['ecpnet', 'ecpnetno']:
        input = (torch.rand(1, 3, 224, 224), torch.rand(1, 1, 4))
    else:
        input = (torch.rand(1, 3, 224, 224),)

    '''[1] using fvcore'''
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count
    print(parameter_count_table(model_ft))

    flops = FlopCountAnalysis(model_ft, input)
    print("FLOPs: {:.2f}G".format(flops.total()/10e9))

    '''[2] using thop'''
    # from thop import profile
    # input = torch.randn(1, 3, 224, 224)
    # macs, params = profile(model_ft, inputs=(input,))
    # print('The model Params: {:.2f}M, MACs: {:.2f}M'.format(params/10e6, macs/10e6))
    '''[3] using torchstat'''
    # from torchstat import stat
    # stat(model_ft, (3, 224, 224))

    '''
    Data augmentation and normalization for training
    Just normalization for validation
    '''
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

    image_datasets = {x: customData(img_path=infile,
                                    txt_path=os.path.join(infile, (x + '.txt')),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

    # wrap your data and label into Tensor
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle = True if x == 'train' else False,
                                                 # num_workers=1
                                                       ) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = torch.nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    '''
    show the training parameters, observe that all parameters are being optimized
    '''
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    '''set optimizer'''
    # optimizer_ft = optim.SGD(params_to_update, lr, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr, weight_decay=0.05)

    '''Setup the loss function'''
    criterion = nn.MSELoss()

    '''
        load Val Ground Truth dataset to compare in every Val epoch
    '''
    val_lab = loadColStr(os.path.join(infile, 'val.txt'), 1)
    _, aVal, bVal = Normalize(val_lab)
    # train_lab = loadColStr(os.path.join(infile, 'train.txt'), 1)
    # _, aVal, bVal = Normalize(train_lab)

    '''Train and evaluate'''
    model_ft, train_hist, val_hist, result, RE_history = train_model(model_ft, dataloaders_dict,
        criterion, optimizer_ft, val_lab, aVal, bVal, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    #############################################
    '''
        Plot training loss
    '''
    plt.figure()
    plt.title("Train and val Loss history vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), train_hist, label="train_hist")
    plt.plot(range(1,num_epochs+1), val_hist, label="val_hist")
    plt.legend()
    # plt.savefig(os.path.join(out_path, 'Hist_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + '.png'))
    plt.show()
    hist = np.vstack((train_hist, val_hist))
    hist_path = os.path.join(out_path, 'Hist_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size))
    # np.savetxt(hist_path, hist.T)
    save_excel(hist.T, hist_path+'.xlsx')

    #############################################
    '''
        plot training and eval loss of all epochs
    '''
    plt.figure()
    plt.title(model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + "Validation Result")
    ts = range(len(val_lab))
    plt.plot(ts, val_lab, label="val_lab")
    plt.plot(ts, result, label="pred_lab")
    plt.legend()
    plt.show()

    res = np.vstack((val_lab, result))
    res_path = os.path.join(out_path, 'Results_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size))
    # np.savetxt(res_path, res.T, fmt='%s')
    save_excel(res.T, res_path + '.xlsx')

    #############################################
    ''' Plot Evaluation result'''
    ### 逆归一化，输出 val 的‘预测的结果’ ###
    # test_lab = loadColStr(os.path.join(infile, 'val.txt'), 1)
    # _, meanVal, stdVal = Normalize(test_lab)
    # result = InvNormalize(result, meanVal, stdVal)


    '''
        layered error
    '''
    print()
    result = np.array(result)
    val_lab = np.array(val_lab)
    RE_history = np.array(RE_history)
    # writer.add_scalars('Validation/result', {'val_lab': val_lab, 'pred_lab': result}, ts)
    print('[Layered error]')
    error = (result - val_lab) / val_lab
    print('Mean(error): {:.2%} | Max(error): {:.2%} | Min(error): {:.2%} | Std(error): {:.2f}'.format(
        np.mean(error), np.max(error), np.min(error), np.std(error)))

    '''RMSE, Mae, R^2'''
    print()
    print('[Statistic error]')
    Rs = mean_squared_error(val_lab, result) ** 0.5
    Mae = mean_absolute_error(val_lab, result)
    R2_s = r2_score(val_lab, result)
    print('RMSE: {:.2f}J | MAE: {:.2f} | R2_s: {:.2f}.'.format(Rs, Mae, R2_s))

    '''total error'''
    print()
    print('[Total error]')
    E1 = np.sum(val_lab)
    E2 = np.sum(result)
    Er = np.abs((E1 - E2) / E1)
    print('GT: {:.2f}J | ECP: {:.2f}J | Er: {:.2%}'.format(E1, E2, Er))


    plt.figure()
    plt.plot(range(args.epochs), RE_history, label="Er history vs. Epoch")
    plt.legend()
    plt.show()
    RE_path = os.path.join(out_path, 'Er_history')
    save_excel(RE_history, RE_path + '.xlsx')

    '''save error to file'''
    res_error = [np.mean(error), np.max(error), np.min(error), np.std(error), Rs, Mae, R2_s, E1, E2, Er*100]
    error_path = os.path.join(out_path, 'Error_' + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size))
    # np.savetxt(error_path, np.array(res_error), fmt='%s')
    save_excel(res_error, error_path+'.xlsx')

    ##########################################################################
    # print("------Test using best trained model------")
    # eval_EC(model_name, model_ft, test_path, infile, args.eval_phase)
