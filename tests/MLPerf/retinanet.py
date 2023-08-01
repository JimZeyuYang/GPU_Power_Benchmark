#!/usr/bin/env python3

import time
import numpy
import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

def main():
    REPEAT = 20

    torch.hub.set_dir('.')

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        exit()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

    bs = 16
    input_tensor = torch.randn((bs, 3, 800, 800)).to(device)

    # Warmup
    with torch.no_grad():  model(input_tensor)
    torch.cuda.synchronize()

    start_ts = int(time.time() * 1_000_000)


    with torch.no_grad():
        for i in range(REPEAT):
            model(input_tensor)

    torch.cuda.synchronize()

    end_ts = int(time.time() * 1_000_000)

    # Store timestamps in a file
    with open("timestamps.csv", "w") as f:
        f.write("timestamp\n")
        f.write(str(start_ts) + "\n")
        f.write(str(end_ts) + "\n")

    # print("Time spent per batch: {:.3f} ms".format((end_ts - start_ts) / 1000 / REPEAT))
    # print("Total runtime: {:.3f} ms".format((end_ts - start_ts) / 1000))


if __name__ == '__main__':
    main()