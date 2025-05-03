from fastai.vision.all import *
from pathlib import Path
import time

from utils.lenet import LeNet
from utils.utils import *  
from fastai.callback.progress import ProgressCallback

if __name__ == '__main__':
    # Generar subsets del 10% al 90%
    fractions = [i / 10 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9

    for idx, frac in enumerate(fractions, start=1):
        pct = int(frac * 100)
        print(f"---------------------------------- {pct}% ---------------------------------------")

        # Fashion-MNIST
        create_random_subset(
            src_path="/mnt/homeGPU/haoweihu/quantize/original/fashion_mnist/train",
            dst_path=f"/mnt/homeGPU/haoweihu/quantize/dataset/{idx:02d}/fashion_mnist/example1",
            train_fraction=frac
        )
        create_random_subset(
            src_path="/mnt/homeGPU/haoweihu/quantize/original/fashion_mnist/train",
            dst_path=f"/mnt/homeGPU/haoweihu/quantize/dataset/{idx:02d}/fashion_mnist/example2",
            train_fraction=frac
        )

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