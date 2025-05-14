import shutil
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from fastai.vision.all import load_learner, PILImage


def extract_all_embeddings(image_paths, learner_path, device='cpu', bs=256):
    """
    Extrae embeddings de la capa fc2 para una lista de imágenes en batch.
    Retorna un ndarray de forma (N, D) con vectores normalizados.
    """
    # Cargar learner y modelo
    learn = load_learner(learner_path)
    model = learn.model.eval().to(device)
    in_c = model.conv1.in_channels

    # Preparar PILImage con canales correctos
    imgs = []
    for p in image_paths:
        pil = Image.open(p)
        pil = pil.convert('L') if in_c == 1 else pil.convert('RGB')
        imgs.append(PILImage.create(pil))

    # Registrar hook en fc2
    feats = []
    def hook_fn(module, inp, out):
        feats.append(out.detach().cpu())
    hook = model.fc2.register_forward_hook(hook_fn)

    # Crear dataloader y forward batched
    dl = learn.dls.test_dl(imgs, bs=bs, num_workers=0)
    with torch.no_grad():
        for batch in dl:
            xb = batch[0].to(device)
            _ = model(xb)
    hook.remove()

    # Concatenar y normalizar
    embs = torch.cat(feats, dim=0).cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-6, None)
    return embs


def fps_embeddings(embs, k):
    """
    Realiza Farthest Point Sampling sobre un array de embeddings.
    Devuelve índices de los k puntos más diversos.
    """
    N, _ = embs.shape
    k = min(k, N)
    idxs = [random.randrange(N)]
    dist_min = np.full(N, np.inf)
    for _ in range(1, k):
        last = embs[idxs[-1]][None, :]
        dist = np.linalg.norm(embs - last, axis=1)
        dist_min = np.minimum(dist_min, dist)
        idxs.append(int(dist_min.argmax()))
    return idxs


def create_deep_subset(src_path, dst_path, train_fraction, valid_ratio=0.2,
                       learner_path=None, device='cpu', bs=256):
    """
    Crea un subset usando deep embeddings (fc2):
      - valid_ratio: % para validación
      - train_fraction: % del resto para training
      - extracción batched y FPS
    """
    if learner_path is None:
        raise ValueError("Se requiere learner_path para create_deep_subset")

    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists(): shutil.rmtree(dst)
    (dst/'training').mkdir(parents=True, exist_ok=True)
    (dst/'valid').mkdir(parents=True, exist_ok=True)

    for class_dir in src.iterdir():
        if not class_dir.is_dir(): continue
        # Listar imágenes
        images = (list(class_dir.glob('**/*.*')) if class_dir.name.lower()=='t-shirt'
                  else list(class_dir.glob('*.*')))
        if not images: continue
        random.shuffle(images)
        total = len(images)
        n_valid = int(total * valid_ratio)
        valid_imgs = images[:n_valid]
        train_imgs = images[n_valid:]

        # Extraer embeddings batched
        embs = extract_all_embeddings(train_imgs, learner_path, device=device, bs=bs)
        n_select = int(len(train_imgs) * train_fraction)
        idxs = fps_embeddings(embs, n_select)
        selected = [train_imgs[i] for i in idxs]

        # Copiar validación
        out_v = dst/'valid'/class_dir.name
        out_v.mkdir(parents=True, exist_ok=True)
        for img in valid_imgs:
            shutil.copy(img, out_v/img.name)
        # Copiar training
        out_t = dst/'training'/class_dir.name
        out_t.mkdir(parents=True, exist_ok=True)
        for img in selected:
            shutil.copy(img, out_t/img.name)

if __name__ == '__main__':
    # Ejemplo: generar todos los subsets deep (10%…90%) para Fashion-MNIST y CIFAR-10
    fractions = [i/10 for i in range(1, 10)]
    base_dir = Path('/mnt/homeGPU/haoweihu/quantize/dataset')
    learner_paths = {
        'fashion_mnist': '/mnt/homeGPU/haoweihu/quantize/models/fmnist_base.pkl'
        #'cifar10':       '/mnt/homeGPU/haoweihu/quantize/models/cifar_base.pkl'
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for idx, frac in enumerate(fractions, start=1):
        pct = int(frac*100)
        print(f"=== Subsets deep al {pct}% ===")

        for ds in ('fashion_mnist',):
            src = f"/mnt/homeGPU/haoweihu/quantize/original/{ds}/train"
            dst = base_dir / f"{idx:02d}" / ds / 'deep'
            create_deep_subset(
                src_path       = src,
                dst_path       = str(dst),
                train_fraction = frac,
                valid_ratio    = 0.2,
                learner_path   = learner_paths[ds],
                device         = device,
                bs             = 256
            )
