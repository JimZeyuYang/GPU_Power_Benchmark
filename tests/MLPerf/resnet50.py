#!/usr/bin/env python3

import time
import torch
import torchvision.models as models

def main():
    torch.hub.set_dir('.')

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        exit()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    model = model.to(device)

    bs = 2048
    input_tensor = torch.randn(bs, 3, 224, 224).to(device)

    with torch.no_grad():
        for i in range(5):
            model(input_tensor)
            torch.cuda.synchronize()
            time.sleep(1)

    torch.cuda.synchronize()


if __name__ == '__main__':
    main()