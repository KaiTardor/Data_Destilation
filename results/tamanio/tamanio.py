import io
import random
from pathlib import Path
from fastai.vision.all import PILImage  # Alternativamente: from PIL import Image

# Se asume que las funciones de transformación están definidas, por ejemplo:
from utils.utils import *

def ver_tamano_25_ejemplos_multitransform(dataset_path, num_samples=25):
    """
    Muestra el total de imágenes del dataset y, de 25 ejemplos aleatorios,
    imprime el tamaño original de cada imagen, el tamaño tras aplicar tres
    transformaciones (umbralizacion_bi, umbralizacion_tri y otsu_threshold) y 
    el porcentaje de reducción obtenido en cada caso.
    
    Args:
        dataset_path (str o Path): Ruta al directorio del dataset.
        num_samples (int): Cantidad de imágenes aleatorias a evaluar (por defecto 25).
    """
    # Definir extensiones válidas
    extensiones_validas = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    dataset_path = Path(dataset_path)
    
    # Obtener la lista de imágenes del dataset
    imagenes = [archivo for archivo in dataset_path.rglob("*") if archivo.suffix.lower() in extensiones_validas]
    total_dataset = len(imagenes)
    print(f"Total de imágenes en el dataset: {total_dataset}\n")
    
    # Seleccionar 25 ejemplos aleatorios (o todos si hay menos)
    if total_dataset > num_samples:
        ejemplos = random.sample(imagenes, num_samples)
    else:
        ejemplos = imagenes
    
    # Inicializar contadores acumulativos
    total_original = 0
    total_bi = 0
    total_tri = 0
    total_otsu = 0

    print("Detalles por imagen (Formato: [Original | Quantizada_BI | Quantizada_TRI | Otsu] y % reducción):")
    for archivo in ejemplos:
        # Obtener el tamaño original en bytes
        tam_original = archivo.stat().st_size
        total_original += tam_original

        # Abrir la imagen
        try:
            imagen_original = PILImage.create(archivo)
        except Exception as e:
            print(f"{archivo.name}: ERROR al abrir la imagen: {e}")
            continue
        
        # Aplicar las tres transformaciones
        try:
            imagen_bi = umbralizacion_bi(imagen_original)
            imagen_tri = umbralizacion_tri(imagen_original)
            imagen_otsu = otsu_threshold_transform(imagen_original)
        except Exception as e:
            print(f"{archivo.name}: ERROR al aplicar las transformaciones: {e}")
            continue
        
        # Guardar las imágenes transformadas en un buffer para medir su tamaño (formato PNG)
        buf_bi = io.BytesIO()
        buf_tri = io.BytesIO()
        buf_otsu = io.BytesIO()
        try:
            imagen_bi.save(buf_bi, format="PNG")
            imagen_tri.save(buf_tri, format="PNG")
            imagen_otsu.save(buf_otsu, format="PNG")
        except Exception as e:
            print(f"{archivo.name}: ERROR al guardar las imágenes transformadas: {e}")
            continue

        tam_bi = buf_bi.tell()
        tam_tri = buf_tri.tell()
        tam_otsu = buf_otsu.tell()

        total_bi += tam_bi
        total_tri += tam_tri
        total_otsu += tam_otsu

        # Calcular el porcentaje de reducción para cada transformación
        reduccion_bi = (1 - (tam_bi / tam_original)) * 100 if tam_original > 0 else 0
        reduccion_tri = (1 - (tam_tri / tam_original)) * 100 if tam_original > 0 else 0
        reduccion_otsu = (1 - (tam_otsu / tam_original)) * 100 if tam_original > 0 else 0

        # Imprimir los detalles para la imagen actual
        print(f"{archivo.name}: {tam_original} bytes | {tam_bi} bytes (BI, -{reduccion_bi:.2f}%) | "
              f"{tam_tri} bytes (TRI, -{reduccion_tri:.2f}%) | {tam_otsu} bytes (Otsu, -{reduccion_otsu:.2f}%)")
    
    # Imprimir resumen acumulado
    print("\nResumen acumulado de los 25 ejemplos:")
    print(f"  Total Original: {total_original} bytes")
    print(f"  Total umbralizacion_bi: {total_bi} bytes")
    print(f"  Total umbralizacion_tri: {total_tri} bytes")
    print(f"  Total otsu_threshold: {total_otsu} bytes")
    if total_original > 0:
        total_reduccion_bi = (1 - (total_bi / total_original)) * 100
        total_reduccion_tri = (1 - (total_tri / total_original)) * 100
        total_reduccion_otsu = (1 - (total_otsu / total_original)) * 100
        print(f"  Reducción total: BI -{total_reduccion_bi:.2f}%, TRI -{total_reduccion_tri:.2f}%, Otsu -{total_reduccion_otsu:.2f}%")
    else:
        print("  No se pudo calcular el total original.")

def ver_tamano_dataset_multitransform(dataset_path):
    """
    Recorre todo el dataset y calcula el tamaño total original y el total tras aplicar 
    las tres transformaciones: umbralizacion_bi, umbralizacion_tri y otsu_threshold_transform.
    Finalmente, muestra el porcentaje de reducción obtenido para cada transformación.

    Args:
        dataset_path (str o Path): Ruta al directorio del dataset.
    """
    # Definir extensiones válidas
    extensiones_validas = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    dataset_path = Path(dataset_path)
    
    # Obtener la lista de imágenes del dataset
    imagenes = [archivo for archivo in dataset_path.rglob("*") if archivo.suffix.lower() in extensiones_validas]
    total_dataset = len(imagenes)
    print(f"Total de imágenes en el dataset: {total_dataset}")
    
    # Inicializar contadores acumulativos y contador de errores
    total_original = 0
    total_bi = 0
    total_tri = 0
    total_otsu = 0
    errores = 0

    for archivo in imagenes:
        try:
            # Obtener el tamaño original en bytes
            tam_original = archivo.stat().st_size
            total_original += tam_original

            # Abrir la imagen
            imagen_original = PILImage.create(archivo)
            
            # Aplicar las tres transformaciones
            imagen_bi = umbralizacion_bi(imagen_original)
            imagen_tri = umbralizacion_tri(imagen_original)
            imagen_otsu = otsu_threshold_transform(imagen_original)
            
            # Guardar las imágenes transformadas en un buffer para medir su tamaño (formato PNG)
            buf_bi = io.BytesIO()
            buf_tri = io.BytesIO()
            buf_otsu = io.BytesIO()
            imagen_bi.save(buf_bi, format="PNG")
            imagen_tri.save(buf_tri, format="PNG")
            imagen_otsu.save(buf_otsu, format="PNG")

            tam_bi = buf_bi.tell()
            tam_tri = buf_tri.tell()
            tam_otsu = buf_otsu.tell()

            total_bi += tam_bi
            total_tri += tam_tri
            total_otsu += tam_otsu
        except Exception as e:
            print(f"Error procesando {archivo.name}: {e}")
            errores += 1
            continue

    # Imprimir resumen acumulado para todo el dataset
    print("\nResumen acumulado del dataset:")
    print(f"  Total Original: {total_original} bytes")
    print(f"  Total umbralizacion_bi: {total_bi} bytes")
    print(f"  Total umbralizacion_tri: {total_tri} bytes")
    print(f"  Total otsu_threshold: {total_otsu} bytes")
    
    if total_original > 0:
        reduccion_bi = (1 - total_bi / total_original) * 100
        reduccion_tri = (1 - total_tri / total_original) * 100
        reduccion_otsu = (1 - total_otsu / total_original) * 100
        print(f"  Porcentaje de reducción: BI -{reduccion_bi:.2f}%, TRI -{reduccion_tri:.2f}%, Otsu -{reduccion_otsu:.2f}%")
    
    if errores:
        print(f"\nSe presentaron {errores} errores al procesar algunas imágenes.")

# Ejemplo de uso:
if __name__ == "__main__":
    # Reemplaza la ruta por la del dataset que deseas analizar
    print(f"---------------------------------------------------------------------------")
    print("--------------------------MNIST---------------------------------------------")
    print(f"---------------------------------------------------------------------------")
    #ver_tamano_25_ejemplos_multitransform("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training")
    ver_tamano_dataset_multitransform("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training")
    print(f"---------------------------------------------------------------------------")
    print("--------------------------FASHION MNIST-------------------------------------")
    print(f"---------------------------------------------------------------------------")
    #ver_tamano_25_ejemplos_multitransform("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train")
    ver_tamano_dataset_multitransform("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train")

    print(f"---------------------------------------------------------------------------")
    print("--------------------------CIFAR---------------------------------------------")
    print(f"---------------------------------------------------------------------------")
    #ver_tamano_25_ejemplos_multitransform("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train")
    ver_tamano_dataset_multitransform("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train")
