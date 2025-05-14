import shutil
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from fastai.vision.all import load_learner, PILImage

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*load_learner` uses Python's insecure pickle module.*"
)

def extract_all_embeddings_efficientnet(image_paths, learner_path, device='cpu', bs=256):
    """
    Extrae embeddings de la última capa lineal para una lista de imágenes en batch.
    Retorna un ndarray de forma (N, D) con vectores normalizados.
    """
    # 1) Cargar learner y modelo
    learn = load_learner(learner_path)
    model = learn.model.eval().to(device)

    # 2) Detectar canales de entrada buscando el primer Conv2d
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No se encontró ningún módulo nn.Conv2d en el modelo")
    in_c = convs[0].in_channels

    # 3) Preparar lista de PILImage con conversión adecuada
    imgs = []
    for p in image_paths:
        pil = Image.open(p)
        pil = pil.convert('L') if in_c == 1 else pil.convert('RGB')
        imgs.append(PILImage.create(pil))

    # 4) Localizar la última capa lineal del modelo
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise RuntimeError("No se encontró ninguna capa nn.Linear en el modelo")
    classifier = linears[-1]

    # 5) Registrar hook para capturar sus salidas
    feats = []
    def hook_fn(module, inp, out):
        feats.append(out.detach().cpu())
    hook = classifier.register_forward_hook(hook_fn)

    # 6) Crear dataloader y forward batched
    dl = learn.dls.test_dl(imgs, bs=bs, num_workers=0)
    with torch.no_grad():
        for batch in dl:
            xb = batch[0].to(device)
            _ = model(xb)
    hook.remove()

    # 7) Concatenar, normalizar y devolver
    embs = torch.cat(feats, dim=0).cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-6, None)
    return embs

from sklearn.cluster import KMeans

def make_bins(embs, n_bins):
    """
    Agrupa embeddings en n_bins clusters con KMeans.
    Devuelve array de etiquetas (len = N).
    """
    km = KMeans(n_clusters=n_bins, random_state=42)
    labels = km.fit_predict(embs)
    return labels

def sample_uniform_bins(image_paths, labels, n_select):
    """
    Selecciona uniformemente n_select imágenes:
     - Calcula cuántos por bin: ceil(n_select / n_bins).
     - Toma esa cantidad de cada bin (o todos si hay menos).
     - Recorta al total exacto n_select.
    """
    bins = np.unique(labels)
    per_bin = int(np.ceil(n_select / len(bins)))
    sel_idxs = []
    for b in bins:
        idxs = np.where(labels == b)[0].tolist()
        if len(idxs) <= per_bin:
            sel_idxs += idxs
        else:
            sel_idxs += random.sample(idxs, per_bin)
    # Recortar al número exacto
    sel_idxs = sel_idxs[:n_select]
    return [image_paths[i] for i in sel_idxs]

def create_dq_subset_efficientnet(
    src_path, dst_path, train_fraction, valid_ratio=0.2,
    learner_path=None, device='cpu', bs=256, n_bins=100
):
    """
    Crea un subset usando Dataset Quantization con embeddings de EfficientNet-V2-Small.
    """
    if learner_path is None:
        raise ValueError("Se requiere learner_path para crear el subset")

    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists(): shutil.rmtree(dst)
    (dst/'training').mkdir(parents=True, exist_ok=True)
    (dst/'valid').mkdir(parents=True, exist_ok=True)

    for class_dir in src.iterdir():
        if not class_dir.is_dir(): continue

        # Cargar todas las imágenes
        imgs = list(class_dir.glob('**/*.*')) if class_dir.name.lower()=='t-shirt' \
               else list(class_dir.glob('*.*'))
        if not imgs: continue

        random.shuffle(imgs)
        total = len(imgs)
        n_valid = int(total * valid_ratio)
        valid_imgs = imgs[:n_valid]
        train_imgs = imgs[n_valid:]
        n_select = int(len(train_imgs) * train_fraction)

        # 1) Extraer embeddings
        embs = extract_all_embeddings_efficientnet(
            train_imgs, learner_path, device=device, bs=bs
        )
        # 2) Crear bins con KMeans
        labels = make_bins(embs, n_bins)
        # 3) Muestreo uniforme
        selected = sample_uniform_bins(train_imgs, labels, n_select)

        # Copiar validación
        out_v = dst/'valid'/class_dir.name
        out_v.mkdir(parents=True, exist_ok=True)
        for f in valid_imgs:
            shutil.copy(f, out_v/f.name)

        # Copiar training
        out_t = dst/'training'/class_dir.name
        out_t.mkdir(parents=True, exist_ok=True)
        for f in selected:
            shutil.copy(f, out_t/f.name)

if __name__ == '__main__':
    # Ejemplo de uso para Fashion-MNIST y CIFAR-10 con EfficientNet-V2-Small
    fractions = [i/10 for i in range(1, 10)]
    base_dir = Path('/mnt/homeGPU/haoweihu/quantize/dataset')
    learner_paths = {
        'fashion_mnist': '/mnt/homeGPU/haoweihu/quantize/efficientnet/models/fmnist_efficientnetv2s.pkl',
        'cifar10':       '/mnt/homeGPU/haoweihu/quantize/efficientnet/models/cifar10_efficientnetv2s.pkl'
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bins_cfg = {'fashion_mnist': 50, 'cifar10': 200}

    for idx, frac in enumerate(fractions, start=1):
        pct = int(frac*100)
        print(f"=== Subsets deep al {pct}% ===")
        for ds in ('fashion_mnist','cifar10'):
            src = f"/mnt/homeGPU/haoweihu/quantize/original/{ds}/train"
            dst = base_dir / f"{idx:02d}" / ds / 'effv2s'
            create_dq_subset_efficientnet(
                src_path       = src,
                dst_path       = str(dst),
                train_fraction = frac,
                valid_ratio    = 0.2,
                learner_path   = learner_paths[ds],
                device         = device,
                bs             = 256,
                n_bins         = bins_cfg[ds]
            )
