import shutil
import time
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, TensorDataset

def extract_all_embeddings_efficientnet_pretrained(
    image_paths, device='cpu', bs=256
):
    """
    Extrae embeddings de la última capa lineal de EfficientNet-V2-Small
    con pesos preentrenados de ImageNet. Retorna un ndarray (N, D), normalizado.
    """
    # 1) Cargar modelo y pipeline de transforms recomendado
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights).eval().to(device)
    tfms = weights.transforms()   # <-- usa el pipeline completo

    # 2) Determinar canales de entrada (sólo para conversión de PIL)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    in_c = convs[0].in_channels if convs else 3

    # 3) Cargar y transformar imágenes
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert('L') if in_c == 1 else Image.open(p).convert('RGB')
        tensors.append(tfms(img))
    ds = TensorDataset(torch.stack(tensors))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    # 4) Hook en la última capa lineal
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    classifier = linears[-1]
    feats = []
    def hook_fn(module, inp, out):
        feats.append(out.detach().cpu())
    hook = classifier.register_forward_hook(hook_fn)

    # 5) Forward pass
    with torch.no_grad():
        for (xb,) in dl:
            _ = model(xb.to(device))

    hook.remove()

    # 6) Concatenar y normalizar
    embs = torch.cat(feats, dim=0).cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-6, None)



def make_bins(embs, n_bins):
    """
    Agrupa embeddings en n_bins clusters con KMeans.
    Devuelve array de etiquetas (len = N).
    """
    km = KMeans(n_clusters=n_bins, random_state=42)
    return km.fit_predict(embs)


def sample_uniform_bins(image_paths, labels, n_select):
    """
    Selecciona uniformemente n_select imágenes:
     - Calcula ceil(n_select / n_bins) por bin.
     - Toma esa cantidad de cada bin (o todos si hay menos).
     - Recorta al total exacto n_select.
    """
    bins = np.unique(labels)
    per_bin = int(np.ceil(n_select / len(bins)))
    sel = []
    for b in bins:
        idxs = np.where(labels == b)[0].tolist()
        sel += idxs if len(idxs) <= per_bin else random.sample(idxs, per_bin)
    sel = sel[:n_select]
    return [image_paths[i] for i in sel]


def create_dq_subset_efficientnet_pretrained(
    src_path, dst_path, train_fraction, valid_ratio=0.2,
    device='cpu', bs=256, n_bins=100
):
    """
    Crea un subset por Dataset Quantization usando embeddings
    de EfficientNet-V2-Small pretrained.
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists(): shutil.rmtree(dst)
    (dst/'training').mkdir(parents=True, exist_ok=True)
    (dst/'valid').mkdir(parents=True, exist_ok=True)

    for class_dir in src.iterdir():
        if not class_dir.is_dir(): continue

        imgs = (list(class_dir.glob('**/*.*')) if class_dir.name.lower()=='t-shirt'
                  else list(class_dir.glob('*.*')))
        if not imgs: continue

        random.shuffle(imgs)
        total = len(imgs)
        n_valid = int(total * valid_ratio)
        valid_imgs = imgs[:n_valid]
        train_imgs = imgs[n_valid:]
        n_select = int(len(train_imgs) * train_fraction)

        # Extraer embeddings
        embs = extract_all_embeddings_efficientnet_pretrained(
            train_imgs, device=device, bs=bs
        )
        # Bins y muestreo
        labels = make_bins(embs, n_bins)
        selected = sample_uniform_bins(train_imgs, labels, n_select)

        # Copiar validación
        out_v = dst/'valid'/class_dir.name
        out_v.mkdir(parents=True, exist_ok=True)
        for f in valid_imgs:
            shutil.copy(f, out_v/f.name)

        # Copiar entrenamiento
        out_t = dst/'training'/class_dir.name
        out_t.mkdir(parents=True, exist_ok=True)
        for f in selected:
            shutil.copy(f, out_t/f.name)


if __name__ == '__main__':
    # Fracciones de subset: 10%, 20%, …, 90%
    fractions = [i/10 for i in range(1, 10)]

    # Carpeta base donde se volcarán los subsets
    base_dir = Path('/mnt/homeGPU/haoweihu/quantize/dataset')

    # Dispositivo GPU/CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Número de bins para cada dataset
    bins_cfg = {
        'fashion_mnist': 50,
        'cifar10':       200
    }

    for idx, frac in enumerate(fractions, start=1):
        pct = int(frac * 100)
        print(f"=== Subsets al {pct}% ===")

        for ds in ('fashion_mnist', ):
            # Origen: tus datos originales de entrenamiento
            src = f"/mnt/homeGPU/haoweihu/quantize/original/{ds}/train"
            dst = base_dir / f"{idx:02d}" / ds / 'effv2s_pretrained'

            t0 = time.perf_counter()
            create_dq_subset_efficientnet_pretrained(
                src_path      = src,
                dst_path      = str(dst),
                train_fraction= frac,
                valid_ratio   = 0.2,
                device        = device,
                bs            = 256,
                n_bins        = bins_cfg[ds]
            )
            elapsed = time.perf_counter() - t0
            print(f"[{ds} al {pct}%] tiempo de creación: {elapsed:.2f} s")
