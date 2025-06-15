import time
import random
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_distances
import cv2
from fastai.vision.core import PILImage
from fastai.vision.all import *
from utils.lenet import LeNet


def extract_embedding(img_path, model, preprocess, device):
    """
    Extrae un embedding normalizado de la imagen usando el modelo LeNet sin la capa final.
    """
    img = Image.open(img_path).convert('L')
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    emb = emb.cpu().numpy().flatten()
    emb /= np.linalg.norm(emb)
    return emb

def media_arit(image_paths):
    """
    Mezcla una lista de imágenes en escala de grises calculando la media aritmética píxel a píxel.
    """
    avg_image = None
    count = 0
    for img_path in image_paths:
        with Image.open(img_path) as img:
            np_img = np.array(img, dtype=np.float32)
            if avg_image is None:
                avg_image = np_img
            else:
                avg_image += np_img
            count += 1
    avg_image /= count
    avg_image = np.clip(avg_image, 0, 255).astype(np.uint8)
    return Image.fromarray(avg_image, mode='L')

def group_by_similarity(images, embeddings, group_size, threshold):
    """
    Agrupa imágenes por similitud usando distancia coseno.

    - images: lista de rutas de imágenes
    - embeddings: dict {img_path: embedding}
    - group_size: tamaño de cada grupo
    """
    remaining = images.copy()
    clusters = []
    while remaining:
        base = remaining.pop(0)
        dists = []
        for other in remaining:
            dist = cosine_distances([embeddings[base]], [embeddings[other]])[0][0]
            if threshold is None or dist <= (1 - threshold):
                dists.append((other, dist))
        dists.sort(key=lambda x: x[1])
        group = [base] + [img for img, _ in dists[: group_size - 1]]
        for img in group[1:]:
            remaining.remove(img)
        clusters.append(group)
    return clusters


def auto_partial_distilled(
    src_path,
    dst_path,
    valid_ratio=0.2,
    distilled_portion=0.8,
    group_fraction=1.0,
    mix_function=media_arit,
    similarity_threshold=0.95
):
    """
    Crea un nuevo dataset aplicando destilación parcial:

    - valid_ratio: proporción de validación.
    - distilled_portion: proporción de entrenamiento a destilar.
    - group_fraction: fracción del tamaño de grupo para mezclar.
    - similarity_threshold: umbral coseno para agrupar duplicados.
    - mix_function: función que mezcla un clúster de imágenes.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cargar modelo LeNet sin capa final
    model = LeNet()
    if hasattr(model, 'fc3'):
        model.fc3 = nn.Identity()
    model.to(device).eval()

    # Preprocesamiento
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists():
        shutil.rmtree(dst)
    train_dst = dst / 'training'
    valid_dst = dst / 'valid'
    train_dst.mkdir(parents=True, exist_ok=True)
    valid_dst.mkdir(parents=True, exist_ok=True)

    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue
        # Caso excepcional para "t-shirt"
        if class_dir.name.lower() == 't-shirt':
            images = list(class_dir.glob('**/*.*'))
        else:
            images = list(class_dir.glob('*.*'))
        if not images:
            continue

        # Barajar y separar en training y validación
        random.shuffle(images)
        total = len(images)
        n_valid = int(total * valid_ratio)
        n_train = total - n_valid

        train_images = images[:n_train]
        valid_images = images[n_train:]

        # Dentro de training, separar el 80% para destilación y el 20% para copiar sin modificar
        n_train_distilled = int(n_train * distilled_portion)
        n_train_original = n_train - n_train_distilled

        distilled_images = train_images[:n_train_distilled]
        original_train_images = train_images[n_train_distilled:]

        # Crear subdirectorios para la clase en training y validación
        new_train_class_dir = train_dst / class_dir.name
        new_train_class_dir.mkdir(parents=True, exist_ok=True)
        new_valid_class_dir = valid_dst / class_dir.name
        new_valid_class_dir.mkdir(parents=True, exist_ok=True)

        # Copiar validación sin cambios
        for img in valid_images:
            shutil.copy(img, new_valid_class_dir / img.name)

        # Destilación
        if distilled_images:
            if group_fraction == 0.0:
                for img in distilled_images:
                    out = mix_function([img])
                    out.save(new_train_class_dir / img.name)
            elif group_fraction == 1.0:
                out = mix_function(distilled_images)
                out.save(new_train_class_dir / 'distilled.png')
            else:
                group_size = max(1, int(len(distilled_images) * group_fraction))
                clusters = group_by_similarity(distilled_images, {
                    img: extract_embedding(img, model, preprocess, device)
                    for img in distilled_images
                }, group_size, threshold=similarity_threshold)
                for idx, cluster in enumerate(clusters):
                    out = mix_function(cluster)
                    out.save(new_train_class_dir / f'distilled_{idx}.png')

        # Copiar originales sin modificar
        for img in original_train_images:
            shutil.copy(img, new_train_class_dir / img.name)

        print(f"Clase {class_dir.name}: {n_train} train, {n_valid} valid.")
