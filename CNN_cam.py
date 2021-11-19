import argparse
import cv2
import numpy as np
import torch
import timm
from torchvision import models
import os
import torch.nn as nn
import timm.models as model
from collections import OrderedDict
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from ecp_utils import initialize_model


def get_args():
    parser = argparse.ArgumentParser()

    # Dataset / Model parameters
    parser.add_argument('--output_dir', metavar='DIR', default='./cam',
                        help='path to output')
    parser.add_argument('--model', default='efficientnet_b3', type=str, metavar='MODEL',
                        help='Name of model to train (default: "resnet18"')
    parser.add_argument('--pth_dir', metavar='DIR', default='./output/lattice_ec',
                        help='path to pth file')

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default="./input_images/A3-5.png",
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='layercam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}

    '''example'''
    # model = models.resnet50(pretrained=True)
    # target_layer = model.layer4[-1]
    # model = models.densenet121(pretrained=True)
    # target_layer = model.features[-1]

    model, image_size = initialize_model(args.model)
    pth_file = os.path.join(args.pth_dir, args.model, 'Best_checkpoint.pth')
    state_dict = torch.load(pth_file)

    """desnet"""
    # model = model.DenseNet121(14)
    # original saved file with DataParallel
    # state_dict = torch.load("/datasets/Dset_Jerry/CXR14/Densenet121_BCE_32/Densenet_8.pkl")  # 模型可以保存为pth文件，也可以为pt文件。
    # create new OrderedDict that does not contain module.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # load params
    model.load_state_dict(new_state_dict)  # 从新加载这个模型。
    model.eval().cuda()

    print(model)
    # 注册hook
    # if args.model == 'squeezenet1_1':
    #     ## For squeezenet1_1
    #     net.features[-1].expand3x3.register_forward_hook(farward_hook)  # 9
    #     net.features[-1].expand3x3.register_backward_hook(backward_hook)
    # elif args.model in ['resnet50', 'resnet152']:
    #     ## For resnet
    #     net.module.layer4[-1].conv3.register_forward_hook(farward_hook)
    #     net.module.layer4[-1].conv3.register_backward_hook(backward_hook)
    # elif args.model in ['efficientnet_b3', 'efficientnet_b4']:
    #     net.module.conv_head.register_forward_hook(farward_hook)
    #     net.module.conv_head.register_backward_hook(backward_hook)

    target_layers = [model.conv_head]

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda)

    rgb_img = cv2.imread(args.image_path, 1)
    rgb_img = np.float32(cv2.resize(rgb_img, (image_size, image_size))) / 255
    input_tensor = preprocess_image(rgb_img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(os.path.join(args.output_dir, args.model, 'cam.jpg'), cam_image)
    # cv2.imwrite(os.path.join(args.output_dir, args.model, 'gb.jpg'), gb)
    # cv2.imwrite(os.path.join(args.output_dir, args.model, 'cam_gb.jpg'), cam_gb)