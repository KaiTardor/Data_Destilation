from fastai.vision.all import *
from pathlib import Path
import time

from utils.lenet import LeNet
from utils.utils import *  
from utils.seleccion import *
from fastai.callback.progress import ProgressCallback

if __name__ == '__main__':
    print(f"----------------------------------DIS 001---------------------------------------")
    
    start_time = time.time()
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_001/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    end_time = time.time()

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_001/mnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    #auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/mnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    
    print(f"\n-------Tiempo mnist: {end_time - start_time:.2f} segundos")
    
    start_time = time.time()
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_001/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    end_time = time.time()

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_001/fmnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    #auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/fmnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    
    print(f"\n--------Tiempo fmnist: {end_time - start_time:.2f} segundos")
    
    start_time = time.time()
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_001/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit_rgb)
    end_time = time.time()

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_001/cifar10/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit_rgb)
    #auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit_rgb)
    
    print(f"\n-----Tiempo cifar: {end_time - start_time:.2f} segundos")
    
    
    print(f"----------------------------------DIS 005---------------------------------------")
    start_time = time.time()
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_005/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)
    end_time = time.time()
    print(f"\n-----Tiempo mnist: {end_time - start_time:.2f} segundos")

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_005/mnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_005/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_005/fmnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_005/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit_rgb)
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_005/cifar10/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit_rgb)
    
    print(f"----------------------------------DIS 0001---------------------------------------")
    start_time = time.time()
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_0001/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)
    end_time = time.time()
    print(f"\n-----Tiempo mnist: {end_time - start_time:.2f} segundos")

    
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_0001/mnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_0001/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_0001/fmnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)

    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_0001/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit_rgb)
    auto_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/auto/distilled_0001/cifar10/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit_rgb)



