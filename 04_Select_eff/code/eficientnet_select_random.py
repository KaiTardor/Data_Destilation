from fastai.vision.all import *
from pathlib import Path
import time

from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from utils.utils import *  
from fastai.callback.progress import ProgressCallback
from fastai.callback.tracker import EarlyStoppingCallback

# Función principal
def main(train_path, test_path, name, use_pretrained = False):
    print("---------------------------------------------------------------------------")
    print(f"Ejecutando: {name}")
    print("---------------------------------------------------------------------------")
    print(use_pretrained)
    # DataBlock para entrenamiento/validación con RandomSplitter 80/20
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='training', valid_name='valid'),
        get_y=parent_label,
        item_tfms=[to_rgb, Resize(224)],
        batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
    )

    # Dataloaders de entrenamiento y validación con batch size 128
    dls = dblock.dataloaders(train_path, bs=64)
    print(f"Número total de imágenes en entrenamiento: {len(dls.train_ds)}")
    print(f"Número total de imágenes en validación: {len(dls.valid_ds)}")
    print("Clases:", dls.vocab)
    
    weights = EfficientNet_V2_S_Weights.DEFAULT if use_pretrained else None
    print("weights =", weights)
    # Learner con EfficientNet V2 Small preentrenado
    learn = vision_learner(
        dls,
        efficientnet_v2_s,
        weights=weights,
        loss_func=CrossEntropyLossFlat(),
        metrics=[accuracy, Recall(average='macro'), F1Score(average='macro')],
        cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=5)]
    )
    learn = learn.to_fp16()
    learn.remove_cb(ProgressCallback)

    # Entrenamiento
    start_time = time.time()
    learn.fit_one_cycle(30)
    elapsed = time.time() - start_time
    print(f"\nTiempo de entrenamiento: {elapsed:.2f} segundos")

    # Validación interna
    loss, acc, recall, f1 = learn.validate()
    print(f"Loss (valid): {loss:.4f}")
    print(f"Accuracy (valid): {acc:.4f}")
    print(f"Recall (valid): {recall:.4f}")
    print(f"F1-Score (valid): {f1:.4f}")

    # Exportar modelo
    out_path = Path('/mnt/homeGPU/haoweihu/quantize/efficientnet/models')/f"{name}.pkl"
    learn.export(out_path)

    # Evaluación en test
    test_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=IndexSplitter([]),
        item_tfms=[to_rgb, Resize(224)],
        batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
    
    test_dls = test_block.dataloaders(test_path, bs=128)
    learn = load_learner(out_path)
    learn.remove_cb(EarlyStoppingCallback)
    
    loss_t, acc_t, recall_t, f1_t = learn.validate(dl=test_dls.train)
    print(f"Loss (test): {loss_t:.4f}")
    print(f"Accuracy (test): {acc_t:.4f}")
    print(f"Recall (test): {recall_t:.4f}")
    print(f"F1-Score (test): {f1_t:.4f}")


if __name__ == '__main__':
    # Directorio base donde tienes los subconjuntos 10%,20%,…,90%
    base_train = Path("/mnt/homeGPU/haoweihu/quantize/dataset")
    #test_path = "/mnt/homeGPU/haoweihu/quantize/original/fashion_mnist/test"
    test_path = "/mnt/homeGPU/haoweihu/quantize/original/cifar10/test"

    # Itera fracciones 0.1…0.9
    fractions = [0.01, 0.03, 0.05, 0.07, 0.09]
    for idx, frac in enumerate(fractions, start=1):
        pct = int(frac * 100)
        #train_subset = base_train/f"{idx:02d}"/"fashion_mnist"/"example1"
        train_subset = base_train/f"{idx:02d}"/"cifar10"/"example1"
        #name = f"fmnist_{pct}pct_effnetv2s"
        name = f"cifar_{pct}pct_effnetv2s_random13579"
        main(train_subset, test_path, name)