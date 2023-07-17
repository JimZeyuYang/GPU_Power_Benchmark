#!/usr/bin/env python3

import time
import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

def main():
    torch.hub.set_dir('.')

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        exit()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

    bs = 64
    input_tensor = torch.randn((bs, 3, 800, 800)).to(device)

    with torch.no_grad():
        model(input_tensor)

    torch.cuda.synchronize()


if __name__ == '__main__':
    main()