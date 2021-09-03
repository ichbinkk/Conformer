""" evalEC
Copyright 2021 Wang Kang
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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# For Conformer
from timm.models import create_model
import models

# output path
out_path = os.path.join('./eval', time.strftime("%Y%m%d-%H%M%S", time.localtime()))


# Number of classes in the dataset
num_classes = 1

lr = 0.001

parm = {}  # 初始化保存模块参数的parm字典

# Set argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='../ECC',
                    help='path to dataset')
parser.add_argument('--dict_dir', metavar='DIR', default='./best_model.pth',
                    help='path to dict')
'''
Setting model and training params, some can use parser to get value.
Models to choose from [resnet, regnet, efficientnet, vit, pit, mixer, deit, swin-vit
alexnet, vgg, squeezenet, densenet, inception, Conformer_tiny_patch16]
'''
parser.add_argument('--model', default='conformer', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet"')
parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-ep', '--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: )')
parser.add_argument('-ft', '--use-pretrained', type=bool, default=True, metavar='N',
                    help='Flag to use fine tuneing(default: False)')
parser.add_argument('-fe', '--feature-extract', type=bool, default=True, metavar='N',
                    help='False to finetune the whole model. True to update the reshaped layer params(default: False)')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# set train and val data prefixs
# prefixs = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2']
prefixs = ['A1','A2','B1','B2','C1','C2', 'D2']
# prefixs = ['A1','B1','C2', 'D2', 'E1']


def eval_model(model, dataloaders, criterion, optimizer, num_epochs=1, is_inception=False):
    since = time.time()
    train_acc_history = []
    val_acc_history = []
    result = []

    model.load_state_dict(torch.load(args.dict_dir))
    # best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = 999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            torch.set_grad_enabled(False)
            outputs = model(inputs)
            if isinstance(outputs, list):
                # Conformer
                for i, o in enumerate(outputs):
                    outputs[i] = o.view(-1)
                loss_list = [criterion(o, labels) / len(outputs) for o in outputs]
                loss = sum(loss_list)
            else:
                outputs = outputs.view(-1)
                loss = criterion(outputs, labels)

            if epoch == num_epochs-1:
                # result.append(outputs)
                if isinstance(outputs, list):
                    # Conformer
                    res = (outputs[0]+outputs[1])/2
                    temp = res.detach().cpu().numpy()
                    for i in range(o.size()[0]):
                        result.append(temp[i])
                else:
                    temp = outputs.detach().cpu().numpy()
                    for i in range(outputs.size()[0]):
                        result.append(temp[i])
                # save loss value
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloaders.dataset)
        val_acc_history.append(epoch_loss)
        print('Eval loss: {:.2f}'.format(epoch_loss))
        print()
    time_elapsed = time.time() - since
    print('Eval complete in time of {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # save best model weights
    # save_path = os.path.join(out_path, 'best_model.pth')
    # torch.save(best_model_wts, save_path)
    return model, train_acc_history, val_acc_history, result


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
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
        model_ft = create_model(
            "Conformer_tiny_patch16",
            pretrained=False,
            num_classes=1,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        for param in model_ft.conv_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        for param in model_ft.trans_cls_head.parameters():
            param.requires_grad = True  # it was require_grad
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = []
            self.img_label = []
            for line in lines:
                img_name = line.strip().split('\t')[0]
                img_prefix = img_name.split('-')[0]
                # if img_prefix is not None:
                if img_prefix in prefixs:
                    self.img_name.append(os.path.join(img_path,img_name))
                    self.img_label.append(float(line.strip().split('\t')[1]))
        ln = len(self.img_label)
        y = self.img_label
        y = torch.tensor(y)
        y = y.numpy()
        print(np.shape(y))
        meanVal = np.mean(y)
        stdVal = np.std(y)
        y = (y - meanVal) / stdVal
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
    return (data*stdVal)+meanVal


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    # get all args params
    args = parser.parse_args()
    infile = args.data_dir
    model_name = args.model
    batch_size = args.batch_size
    use_pretrained = args.use_pretrained
    feature_extract = args.feature_extract

    # output make dir
    out_path = out_path + '_' + model_name
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # get data Dir name
    fn = infile.split('/')[-1]

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)

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

    image_datasets = customData(img_path=infile,
                                    txt_path=os.path.join(infile, ('val.txt')),
                                    data_transforms=data_transforms,
                                    dataset='val')

    # wrap your data and label into Tensor
    dataloaders_dict = torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 # num_workers=1
                                                       )

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized

    # optimizer_ft = optim.SGD(params_to_update, lr, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr, weight_decay=0.05)

    # Setup the loss fxn
    criterion = nn.MSELoss()

    # Evaluate
    model_ft, train_hist, val_hist, result = eval_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=1, is_inception=(model_name=="inception"))

    #######################################################################
    # -----------------------plot and save result-------------------------
    test_lab = loadColStr(os.path.join(infile, 'val.txt'), 1)
    _, meanVal, stdVal = Normalize(test_lab)
    result = InvNormalize(result, meanVal, stdVal)

    ### test ###
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
    ############
    plt.figure()
    plt.title(model_name + "_" + str(lr) + "_" + str(batch_size) + "Validation Result")
    ts = range(len(test_lab))
    plt.plot(ts, test_lab, label="test_lab")
    plt.plot(ts, result, label="pred_lab")
    plt.legend()
    plt.show()
    res = np.vstack((test_lab, result))
    np.savetxt(os.path.join(out_path, 'Eval results of ' + fn + "_" + model_name + "_" + str(batch_size)), res.T, fmt='%s')
    ############
    result = np.array(result)
    test_lab = np.array(test_lab)
    print('[Eval layered error]')
    error = (result - test_lab)/test_lab
    print('Mean(error): {:.2%}.'.format(np.mean(error)))
    print('Max(error): {:.2%}.'.format(np.max(error)))
    print('Min(error): {:.2%}.'.format(np.min(error)))
    print('Std(error): {:.2f}.'.format(np.std(error)))
    Rs = mean_squared_error(test_lab, result) ** 0.5
    Mae = mean_absolute_error(test_lab, result)
    R2_s = r2_score(test_lab, result)
    print('Root mean_squared_error: {:.2f}J, Mean_absolute_error: {:.2f}, R2_score: {:.2f}.'.format(Rs, Mae, R2_s))
    ############
    print('[Eval total models error]')
    E1 = np.sum(test_lab)
    E2 = np.sum(result)
    Er = (E1 - E2)/E2
    print('Actual total EC: {:.2f}J, Predicted total EC: {:.2f}J, Er: {:.2%}'.format(E1,E2,Er))
    res_error = [np.mean(error), np.max(error), np.min(error), np.std(error), E1, E2, Er, Rs, Mae, R2_s]
    np.savetxt(os.path.join(out_path, 'Eval error of ' + fn + "_" + model_name + "_" + str(lr) + "_" + str(batch_size)),
               np.array(res_error), fmt='%s')


