#!/usr/bin/env python3

import time
import numpy
import torch
import torchvision.models as models

def main():
    REPEAT = 32
    SHIFTS = 8

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

    start_ts = []
    end_ts = []

    # Warmup
    with torch.no_grad():  model(input_tensor)
    torch.cuda.synchronize()

    

    with torch.no_grad():
        for i in range(SHIFTS):
            start_ts.append(int(time.time() * 1_000_000))
            for j in range(int(REPEAT/SHIFTS)):
                model(input_tensor)
            torch.cuda.synchronize()
            end_ts.append(int(time.time() * 1_000_000))
            # sleep for 25 miliseconds
            time.sleep(0.025)


    # Store timestamps in a file
    with open("timestamps.csv", "w") as f:
        f.write("timestamp\n")
        for start, end in zip(start_ts, end_ts):
            f.write(str(start) + "\n")
            f.write(str(end) + "\n")

    # print("Time spent per batch: {:.3f} ms".format((end_ts - start_ts) / 1000 / REPEAT))
    # print("Total runtime: {:.3f} ms".format((end_ts - start_ts) / 1000))


if __name__ == '__main__':
    main()