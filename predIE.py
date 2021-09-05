""" PredIE
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
from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count
from torchstat import stat

# For Conformer
from timm.models import create_model
import models

# output path
out_path = os.path.join('./output', time.strftime("%Y%m%d-%H%M%S", time.localtime()))


# Number of classes in the dataset
num_classes = 1

lr = 0.001

parm = {}  # 初始化保存模块参数的parm字典

# Set argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='../ECC',
                    help='path to dataset')
'''
Setting model and training params, some can use parser to get value.
Models to choose from [resnet, regnet, efficientnet, vit, pit, mixer, deit, swin-vit
alexnet, vgg, squeezenet, densenet, inception, Conformer_tiny_patch16, ecpnet]
'''
parser.add_argument('--model', default='ecpnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet"')
parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=2, metavar='N',
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

# set train and val data prefixs
# prefixs = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2']
prefixs = ['A1','A2','B1','B2','C1','C2', 'D2']
# prefixs = ['A1','B1','C2', 'D2', 'E1']


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    result = []

    kernal = np.empty((0, 3))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    min_loss = 999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data 1.
            for inputs, labels, paras in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                paras = paras.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        outputs = outputs.view(-1)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        if args.model == 'ecpnet':
                            outputs = model(inputs, paras)
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

                        # output predicted results
                        if phase == 'val' and epoch == num_epochs - 1:
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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                train_acc_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_loss)
                if epoch_loss < min_loss:
                    min_loss = epoch_loss  # update min_loss
                    print('Best val min_loss: {:4f} in Epoch {}/{}'.format(min_loss, epoch, num_epochs - 1))
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
        print()
    time_elapsed = time.time() - since
    print('Training complete in time of {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save best model weights
    # output make dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, args.model + '_' + str(best_epoch) + ' in ' + str(num_epochs) + '_' + 'best_model.pth')
    torch.save(best_model_wts, save_path)

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
        """
        Conformer_tiny_patch16, Conformer_small_patch16, Conformer_small_patch32, Conformer_base_patch16
        """
        model_ft = create_model(
            "Conformer_base_patch16",
            pretrained=use_pretrained,
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


    elif model_name == "ecpnet":
        model_ft = create_model(
            "EcpNet_tiny_patch16",
            pretrained=use_pretrained,
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
        for param in model_ft.mlp_cls_head.parameters():
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
    data = torch.tensor(data)
    data= data.numpy()
    res = data
    # [1] mean-std norm
    # if len(np.shape(data)) == 1:
    #     aVal = np.mean(data)
    #     bVal = np.std(data)
    #     res = (data-aVal)/bVal
    # else:
    #     for i in [2,3]:
    #         aVal = np.mean(data[:,i])
    #         bVal = np.std(data[:,i])
    #         res[:, i] = (data[:, i]-aVal)/bVal
    # [2] 0-1 norm
    if len(np.shape(data)) == 1:
        aVal = np.min(data)
        bVal = np.max(data)
        res = (data-aVal)/(bVal-aVal)
    else:
        for i in [2,3]:
            aVal = np.min(data[:,i])
            bVal = np.max(data[:,i])
            res[:, i] = (data[:, i]-aVal)/(bVal-aVal)
    bVal = bVal-aVal
    return res, aVal, bVal


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
    num_epochs = args.epochs
    use_pretrained = args.use_pretrained
    feature_extract = args.feature_extract
    out_path = out_path + '_' + args.model

    # get data Dir name
    fn = infile.split('/')[-1]

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)

    # Analyze flops and params
    print(parameter_count_table(model_ft))
    if args.model == 'ecpnet':
        input = (torch.rand(1, 3, 224, 224), torch.rand(1, 1, 4))
    else:
        input = (torch.rand(1, 3, 224, 224),)
    flops = FlopCountAnalysis(model_ft, input)
    print("FLOPs: ", flops.total())
    print('-'*30)
    # stat(model_ft, (3, 224, 224))

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
                                                 shuffle = True if x == 'train' else False,
                                                 # num_workers=1
                                                       ) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # show the training parameters
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

    # Train and evaluate
    model_ft, train_hist, val_hist, result = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    plt.figure()
    plt.title("Train and val Loss history vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), train_hist, label="train_hist")
    plt.plot(range(1,num_epochs+1), val_hist, label="val_hist")
    # plt.ylim((0, 2.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(os.path.join(out_path, 'Hist of ' + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + '.png'))
    plt.show()
    hist = np.vstack((train_hist, val_hist))
    np.savetxt(os.path.join(out_path, "Hist of " + fn + "_" + str(model_name) + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size)), hist.T)

    #######################################################################
    # -----------------------Evaluation--------------------------------
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
    plt.title(model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size) + "Validation Result")
    ts = range(len(test_lab))
    plt.plot(ts, test_lab, label="test_lab")
    plt.plot(ts, result, label="pred_lab")
    plt.legend()
    plt.show()
    res = np.vstack((test_lab, result))
    np.savetxt(os.path.join(out_path, 'Results of ' + fn + "_" + model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size)), res.T, fmt='%s')
    ############
    result = np.array(result)
    test_lab = np.array(test_lab)
    print('[Layered error]')
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
    print('[Total models error]')
    E1 = np.sum(test_lab)
    E2 = np.sum(result)
    Er = (E1 - E2)/E2
    print('Actual total EC: {:.2f}J, Predicted total EC: {:.2f}J, Er: {:.2%}'.format(E1,E2,Er))
    res_error = [np.mean(error), np.max(error), np.min(error), np.std(error), Rs, Mae, R2_s, E1, E2, Er]
    np.savetxt(os.path.join(out_path, 'Error of ' + fn + "_" + model_name + "_" + str(num_epochs) + "_" + str(lr) + "_" + str(batch_size)),
               np.array(res_error), fmt='%s')


