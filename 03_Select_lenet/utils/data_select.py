from pathlib import Path
from utils.seleccion_clasica import *

if __name__ == '__main__':
    methods  = ['phash', 'fps', 'kmeans']
    fractions = [i/10 for i in range(1, 10)]  # 0.1, 0.2, …, 0.9
    base_dir  = Path('/mnt/homeGPU/haoweihu/quantize/dataset')

    for idx, frac in enumerate(fractions, start=1):
        pct = int(frac * 100)

        # Fashion-MNIST
        src_fm = '/mnt/homeGPU/haoweihu/quantize/original/fashion_mnist/train'
        for method in methods:
            dst_fm = base_dir / f"{idx:02d}" / "fashion_mnist" / method
            create_diverse_subset(
                src_path        = src_fm,
                dst_path        = str(dst_fm),
                train_fraction  = frac,
                valid_ratio     = 0.2,
                method          = method
            )

        # CIFAR-10
        src_ci = '/mnt/homeGPU/haoweihu/quantize/original/cifar10/train'
        for method in methods:
            dst_ci = base_dir / f"{idx:02d}" / "cifar10" / method
            create_diverse_subset(
                src_path        = src_ci,
                dst_path        = str(dst_ci),
                train_fraction  = frac,
                valid_ratio     = 0.2,
                method          = method
            )
