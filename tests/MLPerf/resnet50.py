#!/usr/bin/env python3

import time
import numpy
import torch
import torchvision.models as models

def main():
    REPEAT = 4

    torch.hub.set_dir('.')

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        exit()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    model = model.to(device)

    bs = 1024
    input_tensor = torch.randn(bs, 3, 224, 224).to(device)

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