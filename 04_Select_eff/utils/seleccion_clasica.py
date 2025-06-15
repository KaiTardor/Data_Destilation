import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import imagehash


def get_color_hist(image, bins=8):
    """
    Calcula un histograma de color RGB concatenado y lo normaliza.
    Vector de tamaño bins*3.
    """
    arr = np.array(image.convert('RGB'), dtype=np.uint8)
    hist = []
    for c in range(3):
        channel = arr[..., c]
        h, _ = np.histogram(channel, bins=bins, range=(0, 256))
        hist.append(h)
    hist = np.concatenate(hist).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def get_gray_hist(image, bins=8):
    """
    Calcula un histograma de escala de grises y lo normaliza.
    Vector de tamaño bins.
    """
    arr = np.array(image.convert('L'), dtype=np.uint8)
    h, _ = np.histogram(arr, bins=bins, range=(0, 256))
    hist = h.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def select_diverse_kmeans(image_paths, k, bins=8, random_state=42):
    """
    Selección diversa mediante KMeans sobre histogramas adaptados al modo de la imagen.
    """
    feats = []
    modes = []
    # Construir vectores de características según modo
    for p in image_paths:
        with Image.open(p) as img:
            if img.mode == 'L':
                feats.append(get_gray_hist(img, bins))
                modes.append('gray')
            else:
                feats.append(get_color_hist(img, bins))
                modes.append('rgb')
    feats = np.stack(feats)
    k = min(k, len(image_paths))
    km = KMeans(n_clusters=k, random_state=random_state).fit(feats)
    labels = km.labels_
    centers = km.cluster_centers_
    selection = []
    for i in range(k):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            continue
        # Distancia al centroide
        dists = np.linalg.norm(feats[idxs] - centers[i], axis=1)
        best = idxs[np.argmin(dists)]
        selection.append(image_paths[best])
    return selection


def select_diverse_phash(image_paths, k):
    """
    Selección diversa usando perceptual hash (pHash) y clustering sobre bits.
    Funciona tanto en RGB como en L (phash los convierte internamente).
    """
    hashes = []
    for p in image_paths:
        img = Image.open(p)
        hashes.append(imagehash.phash(img))
    bits = np.array([h.hash.flatten().astype(int) for h in hashes])
    k = min(k, len(image_paths))
    km = KMeans(n_clusters=k).fit(bits)
    labels = km.labels_
    centers = km.cluster_centers_
    selection = []
    for i in range(k):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            continue
        dists = np.linalg.norm(bits[idxs] - centers[i], axis=1)
        best = idxs[np.argmin(dists)]
        selection.append(image_paths[best])
    return selection


def select_diverse_fps(image_paths, k, bins=8):
    """
    Selección diversa por Farthest Point Sampling usando histograma adaptado.
    """
    # Extraer características adaptadas al modo
    feats = []
    for p in image_paths:
        img = Image.open(p)
        if img.mode == 'L':
            feats.append(get_gray_hist(img, bins))
        else:
            feats.append(get_color_hist(img, bins))
    feats = np.stack(feats)
    N = len(image_paths)
    k = min(k, N)
    # Iniciar con punto aleatorio
    idx0 = random.randrange(N)
    selected = [idx0]
    distances = np.full(N, np.inf)
    for _ in range(1, k):
        dist_to_sel = np.linalg.norm(feats - feats[selected[-1]], axis=1)
        distances = np.minimum(distances, dist_to_sel)
        next_idx = np.argmax(distances)
        selected.append(next_idx)
    return [image_paths[i] for i in selected]


def create_diverse_subset(src_path, dst_path, train_fraction, valid_ratio=0.2, 
                          method='kmeans', **kwargs):
    """
    Crea un subset diverso del dataset, soportando imágenes en color y grayscale.

    Parámetros:
      src_path: carpeta origen con subcarpetas por clase.
      dst_path: carpeta destino ('training' y 'valid').
      train_fraction: fracción de train a incluir.
      valid_ratio: fracción fija para validación.
      method: 'random', 'kmeans', 'phash' o 'fps'.
      kwargs: parámetros adicionales según método.
    """
    selector = {
        'random': lambda paths, k, **kw: random.sample(paths, k),
        'kmeans': select_diverse_kmeans,
        'phash': select_diverse_phash,
        'fps': select_diverse_fps
    }.get(method)
    if selector is None:
        raise ValueError(f"Método desconocido: {method}")

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if dst_path.exists(): shutil.rmtree(dst_path)
    (dst_path/'training').mkdir(parents=True, exist_ok=True)
    (dst_path/'valid').mkdir(parents=True, exist_ok=True)

    for class_dir in src_path.iterdir():
        if not class_dir.is_dir(): continue
        images = (list(class_dir.glob('**/*.*')) if class_dir.name.lower()=='t-shirt'
                  else list(class_dir.glob('*.*')))
        if not images: continue
        random.shuffle(images)
        total = len(images)
        n_valid = int(total * valid_ratio)
        valid_images = images[:n_valid]
        train_images = images[n_valid:]
        n_train_sel = int(len(train_images) * train_fraction)
        train_sel = selector(train_images, n_train_sel, **kwargs)
        # Crear dirs y copiar
        for subset, imgs in [('valid', valid_images), ('training', train_sel)]:
            out_dir = dst_path/subset/class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in imgs:
                shutil.copy(p, out_dir/p.name)
