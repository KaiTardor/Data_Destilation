#!/usr/bin/env python3
import shutil, random
from pathlib import Path

def create_stratified_subset(src_path, dst_path,
                             valid_ratio=0.2,
                             train_fraction=1.0,
                             seed=42):
    """
    Crea train/valid estratificados en puro Python:
     - valid_ratio: fracción de cada clase para validación.
     - train_fraction: fracción adicional (del resto) para train.
     - seed: semilla para reproducibilidad.
    Maneja el caso especial de 't-shirt' con glob recursivo.
    """
    random.seed(seed)
    src, dst = Path(src_path), Path(dst_path)
    # 1) Preparo destino
    if dst.exists(): 
        shutil.rmtree(dst)
    (dst/"training").mkdir(parents=True, exist_ok=True)
    (dst/"valid").mkdir(parents=True, exist_ok=True)

    # 2) Por cada clase…
    for class_dir in src.iterdir():
        if not class_dir.is_dir(): 
            continue

        # glob especial para t-shirt
        if class_dir.name.lower() == "t-shirt":
            images = list(class_dir.glob("**/*.*"))
        else:
            images = list(class_dir.glob("*.*"))

        total = len(images)
        if total == 0:
            print(f"¡Aviso!: no hay imágenes en {class_dir}")
            continue

        # 3) Barajamos y separamos valid
        random.shuffle(images)
        n_valid = int(total * valid_ratio)
        valid_images = images[:n_valid]
        train_pool    = images[n_valid:]

        # 4) Submuestreo adicional del train_pool
        if train_fraction < 1.0:
            n_train_sel = int(len(train_pool) * train_fraction)
            train_images = random.sample(train_pool, n_train_sel)
        else:
            train_images = train_pool

        # 5) Copiar a destino
        for split, imgs in [("valid", valid_images),
                            ("training", train_images)]:
            for img in imgs:
                dest_dir = dst/split/class_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(img, dest_dir/img.name)

if __name__ == "__main__":
    VALID_RATIO    = 0.2
    SEED           = 42
    BASE_SRC_FMN   = "/mnt/homeGPU/haoweihu/quantize/original/fashion_mnist/train"
    BASE_SRC_CIFAR = "/mnt/homeGPU/haoweihu/quantize/original/cifar10/train"
    BASE_DST       = "/mnt/homeGPU/haoweihu/quantize/dataset"

    # Iterar fracciones 10%, 20%, …, 90%
    for idx, frac in enumerate([i/10 for i in range(1,10)], start=1):
        pct = int(frac * 100)
        print(f"=== Generando {pct}% train (seed={SEED}) ===")

        # Fashion-MNIST
        dst_fmn = Path(BASE_DST)/f"{idx:02d}"/"fashion_mnist"/"stratified"
        create_stratified_subset(
            src_path=BASE_SRC_FMN,
            dst_path=dst_fmn,
            valid_ratio=VALID_RATIO,
            train_fraction=frac,
            seed=SEED
        )

        # CIFAR-10
        dst_cifar = Path(BASE_DST)/f"{idx:02d}"/"cifar10"/"stratified"
        create_stratified_subset(
            src_path=BASE_SRC_CIFAR,
            dst_path=dst_cifar,
            valid_ratio=VALID_RATIO,
            train_fraction=frac,
            seed=SEED
        )

    print("¡Conjuntos generados satisfactoriamente!")
