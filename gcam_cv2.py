import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json


import argparse
from ECmodels import *

# Set argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--path_img', metavar='DIR', default='./input_images/A3-5.png',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./cam',
                    help='path to output')
parser.add_argument('--model', default='efficientnet_b3', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet18"')
parser.add_argument('--pth_dir', metavar='DIR', default='./output/lattice_ec',
                    help='path to pth file')


# 图片预处理
def img_preprocess(img_in,image_size):
    img = img_in.copy()
    img = img[:, :, ::-1]   				# 1

    # transform = transforms.Compose([
    #     transforms.Resize([image_size, image_size])
    # ])
    # img = transform(img)

    img = np.ascontiguousarray(img)			# 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)					# 3
    return img

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

# 计算grad-cam并可视化
def gcam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.abs(cam)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    cam_img = 0.5 * heatmap + 0.5 * img
    # cam_img = 1 * heatmap + 0 * img

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path_cam_img = os.path.join(out_dir, "gcam.jpg")

    cv2.imwrite(path_cam_img, cam_img)

    return cam_img

# 计算 feature_map
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += feature_map[i, :, :]							# 7
    cam = np.abs(cam)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    cam_img = 0.5 * heatmap + 0.5 * img
    # cam_img = 1 * heatmap + 0 * img

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path_cam_img = os.path.join(out_dir, "fmp.jpg")

    cv2.imwrite(path_cam_img, cam_img)

    return cam_img


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    # get all args params
    args = parser.parse_args()

    path_img = args.path_img
    output_dir = os.path.join(args.output_dir, args.model)

    '''assign one image directly'''
    # path_img = './cam/bicycle.jpg'
    # json_path = './cam/labels.json'
    # with open(json_path, 'r') as load_f:
    #     load_json = json.load(load_f)
    # classes = {int(key): value for (key, value)
    #            in load_json.items()}
    ## 只取标签名
    # classes = list(classes.get(key) for key in range(1000))

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)
    # img = Image.open(path_img).convert('RGB')


    '''load model'''
    ### method (1)
    # net = models.squeezenet1_1(pretrained=True)
    # pthfile = './pretrained/squeezenet1_1-f364aa15.pth'
    # net.load_state_dict(torch.load(pthfile))

    ### method (2)
    pth_file = os.path.join(args.pth_dir, args.model, 'Best_checkpoint.pth')
    if os.path.exists(pth_file):
        net, image_size = initialize_model(args.model)
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(pth_file))
    else:
        print('the *.pth file does not exist')

    net.eval()  # 8
    print(net)

    # 注册hook
    if args.model == 'squeezenet1_1':
        ## For squeezenet1_1
        net.features[-1].expand3x3.register_forward_hook(farward_hook)  # 9
        net.features[-1].expand3x3.register_backward_hook(backward_hook)
    elif args.model in ['resnet50', 'resnet152']:
        ## For resnet
        net.module.layer4[-1].conv3.register_forward_hook(farward_hook)
        net.module.layer4[-1].conv3.register_backward_hook(backward_hook)
    elif args.model in ['efficientnet_b3', 'efficientnet_b4']:
        net.module.conv_head.register_forward_hook(farward_hook)
        net.module.conv_head.register_backward_hook(backward_hook)

    # forward
    img = cv2.resize(img, (image_size, image_size))
    img_input = img_preprocess(img, image_size)

    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    # print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    # generate grads and feature map
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # show image
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # cv2.imread(os.path.join(output_dir, 'gcam.jpg'))

    # save GCAM
    gcam_show_img(img, fmap, grads_val, output_dir)
    # save FMP
    cam_show_img(img, fmap, grads_val, output_dir)

