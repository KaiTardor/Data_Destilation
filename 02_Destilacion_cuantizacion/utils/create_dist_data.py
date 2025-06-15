from fastai.vision.all import *
from pathlib import Path
import time

from utils.lenet import LeNet
from utils.utils import *  
from fastai.callback.progress import ProgressCallback

if __name__ == '__main__':
    print(f"----------------------------------DIS 001---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/mnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/mnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)

    print(f"----------------------------------DIS 001---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/fmnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/fmnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit)
    
    print(f"----------------------------------DIS 001---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.01, mix_function=media_arit_rgb)
    
    
    print(f"----------------------------------DIS 005---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/mnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/mnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)

    print(f"----------------------------------DIS 005---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/fmnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/fmnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit)
    
    print(f"----------------------------------DIS 005---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.05, mix_function=media_arit_rgb)
    
    
    print(f"----------------------------------DIS 0001---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/mnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/mnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)

    print(f"----------------------------------DIS 0001---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/fmnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/fmnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit)
    
    print(f"----------------------------------DIS 0001---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.001, mix_function=media_arit_rgb)
    
    #print(f"----------------------------------DIS 01---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/mnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/mnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit)

    #print(f"----------------------------------DIS 01---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/fmnist/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/fmnist/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit)
    
    #print(f"----------------------------------DIS 01---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/cifar10/example2", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit_rgb)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_01/cifar10/example3", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.1, mix_function=media_arit_rgb)
    
    #print(f"----------------------------------DIS 02---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_02/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.2, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_02/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.2, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_02/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.2, mix_function=media_arit_rgb)
    
    #print(f"----------------------------------DIS 04---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_04/mnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.4, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_04/fmnist/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.4, mix_function=media_arit)
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_04/cifar10/example1", valid_ratio=0.2, distilled_portion=0.8, group_fraction=0.4, mix_function=media_arit_rgb)

    print(f"----------------------------------DIS 99---------------------------------------")
    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/mnist/example1", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit)
    create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/mnist/example2", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit)
    create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/mnist/example3", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit)

    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/fmnist/example1", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit)
    create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/fmnist/example2", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit)
    create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/fmnist/example3", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit)

    #create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/cifar10/example1", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit_rgb)
    create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/cifar10/example2", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit_rgb)
    create_partial_distilled("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/distilled_99/cifar10/example3", valid_ratio=0.2, distilled_portion=0.99, group_fraction=0.001, mix_function=media_arit_rgb)

    