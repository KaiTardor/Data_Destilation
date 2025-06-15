from fastai.vision.all import *
from pathlib import Path
import time

from utils.lenet import LeNet
from utils.utils import *  
from fastai.callback.progress import ProgressCallback

if __name__ == '__main__':
    # Generar subsets del 10% al 90%
    fractions = [0.01, 0.03, 0.05, 0.07, 0.09]
    
    for idx, frac in enumerate(fractions, start=1):
        pct = int(frac * 100)
        print(f"---------------------------------- {pct}% ---------------------------------------")

        # CIFAR-10
        create_random_subset(
            src_path="/mnt/homeGPU/haoweihu/quantize/original/cifar10/train",
            dst_path=f"/mnt/homeGPU/haoweihu/quantize/dataset/{idx:02d}/cifar10/example1",
            train_fraction=frac
        )
        create_random_subset(
            src_path="/mnt/homeGPU/haoweihu/quantize/original/cifar10/train",
            dst_path=f"/mnt/homeGPU/haoweihu/quantize/dataset/{idx:02d}/cifar10/example2",
            train_fraction=frac
        )
        create_random_subset(
            src_path="/mnt/homeGPU/haoweihu/quantize/original/cifar10/train",
            dst_path=f"/mnt/homeGPU/haoweihu/quantize/dataset/{idx:02d}/cifar10/example3",
            train_fraction=frac
        )