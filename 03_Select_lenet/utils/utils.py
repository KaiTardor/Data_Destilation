import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random
import shutil
from pathlib import Path
from PIL import Image
from fastai.vision.core import PILImage
from fastai.vision.all import *
    
def media_arit(image_paths):
    avg_image = None
    count = 0
    # Sumar todas las imágenes (convertidas a array NumPy)
    for img_path in image_paths:
        with Image.open(img_path) as img:
            np_img = np.array(img, dtype=np.float32)
            if avg_image is None:
                avg_image = np_img
            else:
                avg_image += np_img
            count += 1
    # Calcular la media y convertir a uint8
    avg_image /= count
    avg_image = np.clip(avg_image, 0, 255).astype(np.uint8)
    return Image.fromarray(avg_image, mode='L')

def media_arit_rgb(image_paths):
    avg_image = None
    count = 0
    # Sumar todas las imágenes (convertidas a array NumPy sin modificar)
    for img_path in image_paths:
        with Image.open(img_path) as img:
            # Asegúrate de que la imagen esté en RGB
            img = img.convert('RGB')
            np_img = np.array(img, dtype=np.float32)
            if avg_image is None:
                avg_image = np_img
            else:
                avg_image += np_img
            count += 1
    # Calcular la media y convertir a uint8
    avg_image /= count
    avg_image = np.clip(avg_image, 0, 255).astype(np.uint8)
    return Image.fromarray(avg_image, mode='RGB')


def create_partial_distilled(src_path, dst_path, valid_ratio=0.2, distilled_portion=0.8, group_fraction=1, mix_function=media_arit):
    """
    Crea un nuevo dataset a partir de src_path con la siguiente estrategia:

      - Se divide en 80% training y 20% validación (valid_ratio).
      - En el conjunto de training:
          * El 80% de las imágenes se procesan mediante destilación usando mix_function.
          * El 20% restante se copia sin modificar.
      - En el conjunto de validación se copian las imágenes sin modificaciones.

    Parámetros:
      src_path: Ruta a la carpeta original, que debe tener subcarpetas para cada clase.
      dst_path: Ruta destino para el nuevo dataset.
      valid_ratio: Proporción de imágenes para validación (ej. 0.2 para 20%).
      distilled_portion: Proporción de imágenes dentro del training que serán destiladas (ej. 0.8 para 80%).
      group_fraction:
          * Si es 1, se mezclan todas las imágenes del subconjunto de destilación en una sola imagen.
          * Si es un valor entre 0 y 1, se agrupan en bloques cuyo tamaño es group_size = int(len(distilled_images) * group_fraction).
          * Si es 0, se procesa cada imagen individualmente (aunque normalmente querrás usar 1 o un valor intermedio).
      mix_function: Función que recibe una lista de rutas de imágenes y devuelve una imagen destilada.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # Eliminar dst_path si ya existe
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Crear directorios para training y validación
    train_dst = dst_path / "training"
    valid_dst = dst_path / "valid"
    train_dst.mkdir(parents=True, exist_ok=True)
    valid_dst.mkdir(parents=True, exist_ok=True)

    # Procesar cada subcarpeta (clase) en src_path
    for class_dir in src_path.iterdir():
        if not class_dir.is_dir():
            continue

        if class_dir.name.lower() == "t-shirt":
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

        # Procesar las imágenes para destilación (80% del conjunto de training)
        if group_fraction == 0:
            # Procesar cada imagen individualmente (aunque esto normalmente no es lo esperado)
            for img_path in distilled_images:
                distilled_img = mix_function([img_path])
                # Se guarda con el mismo nombre original
                new_file = new_train_class_dir / img_path.name
                distilled_img.save(new_file)
        elif group_fraction == 1:
            # Mezclar todas las imágenes del subconjunto en una sola imagen destilada
            distilled_img = mix_function(distilled_images)
            distilled_filename = new_train_class_dir / "distilled.png"
            distilled_img.save(distilled_filename)
        else:
            # Agrupar las imágenes en bloques
            group_size = max(1, int(len(distilled_images) * group_fraction))
            group_count = 0
            for i in range(0, len(distilled_images), group_size):
                group = distilled_images[i:i + group_size]
                distilled_img = mix_function(group)
                distilled_filename = new_train_class_dir / f"distilled_{group_count}.png"
                distilled_img.save(distilled_filename)
                group_count += 1

        # Copiar las imágenes originales (20% de training) sin modificar
        for img_path in original_train_images:
            shutil.copy(img_path, new_train_class_dir)

        # En validación se copian todas las imágenes sin procesar
        for img_path in valid_images:
            shutil.copy(img_path, new_valid_class_dir)


def umbralizacion_tri(img: PILImage):
    return img.quantize(colors=3)

def umbralizacion_bi(img: PILImage):
    return img.quantize(colors=2)

import cv2

def otsu_threshold_transform(img: PILImage):
    img_gray = img.convert("L")
    img_array = np.array(img_gray)
    _, thresholded = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return PILImage.create(thresholded)

from PIL import Image

def to_rgb(x):
    if hasattr(x, 'convert'):
        if x.mode != 'RGB':
            x = x.convert('RGB')
    return x

def get_filtered_images(path):
    return [img for img in get_image_files(path) if not img.name.startswith("distilled_")]

def get_only_distilled_images(path):
    imgs = get_image_files(path)
    return [
        img for img in imgs
        if img.parent.parent.name != 'training'   # si NO es training → mantenla
        or img.name.startswith('distilled_')      # si es training → solo si empieza por 'distilled_'
    ]

import shutil
import random
from pathlib import Path

def create_random_subset(src_path, dst_path, train_fraction, valid_ratio=0.2):
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # Si ya existía, lo borramos entero
    if dst_path.exists():
        shutil.rmtree(dst_path)
    (dst_path / "training").mkdir(parents=True)
    (dst_path / "valid").mkdir(parents=True)

    for class_dir in src_path.iterdir():
        if not class_dir.is_dir():
            continue

        if class_dir.name.lower() == "t-shirt":
            images = list(class_dir.glob('**/*.*'))
        else:
            images = list(class_dir.glob('*.*'))
        if not images:
            continue

        random.shuffle(images)
        total = len(images)
        n_valid = int(total * valid_ratio)
        valid_images = images[:n_valid]
        train_images = images[n_valid:]

        # Ahora muestreamos train_fraction del resto
        n_train_sel = int(len(train_images) * train_fraction)
        train_sel = random.sample(train_images, n_train_sel)

        # Creamos carpetas de destino
        train_dest = dst_path / "training" / class_dir.name
        valid_dest = dst_path / "valid" / class_dir.name
        train_dest.mkdir(parents=True, exist_ok=True)
        valid_dest.mkdir(parents=True, exist_ok=True)

        # Copiamos valid
        for img in valid_images:
            shutil.copy(img, valid_dest / img.name)

        # Copiamos train muestreado
        for img in train_sel:
            shutil.copy(img, train_dest / img.name)

