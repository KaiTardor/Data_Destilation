from fastai.vision.all import *
from pathlib import Path
import time

from utils.lenet import LeNet, LeNet2
from utils.utils import *  
from fastai.callback.progress import ProgressCallback


def main(train_path, name):
    print(f"\n---------------------------------------------------------------------------")
    print(f"Ejecutando: {name}")
    print(f"---------------------------------------------------------------------------\n")

    # Crear el DataBlock para entrenamiento y validación
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='training', valid_name='valid'),
        get_y=parent_label,
        item_tfms=[umbralizacion_bi]
    )

    dls = dblock.dataloaders(train_path, bs=64)

    # Obtener el número total de imágenes
    num_train = len(dls.train_ds)
    num_valid = len(dls.valid_ds)

    print(f"Número total de imágenes en entrenamiento: {num_train}")
    print(f"Número total de imágenes en validación: {num_valid}")

    # Obtener la distribución de clases
    class_counts = dls.train_ds.vocab  # Obtiene las clases
    print("\nClases:", class_counts)

    # Inicializar el modelo LeNet con 10 clases
    model = LeNet2(num_classes=10)
    
    # Crear el objeto Learner
    learn = Learner(
        dls, 
        model, 
        loss_func=CrossEntropyLossFlat(), 
        metrics=[accuracy, Recall(average='macro'), F1Score(average='macro')], 
        cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=5)]
    )

    # Entrenamiento del modelo con fit_one_cycle
    learn.remove_cb(ProgressCallback)

    start_time = time.time()
    learn.fit_one_cycle(30)
    end_time = time.time()

    print(f"\nTiempo de entrenamiento: {end_time - start_time:.2f} segundos")

    # Validación y reporte de métricas
    loss, acc, recall, f1 = learn.validate()
    print(f'Loss: {loss:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f"Recall: {recall:.4f}")
    print(f'F1-Score: {f1:.4f}')

    # Exportar el modelo entrenado
    learn.export('/mnt/homeGPU/haoweihu/code/models/'+name+'.pkl')

    # Crear el DataBlock para los datos de prueba
    test_block = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=IndexSplitter([]),
        item_tfms=[umbralizacion_bi]
    )

    test_dls = test_block.dataloaders("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test")

    # Cargar el modelo previamente exportado y remover el callback de EarlyStopping
    learn = load_learner('/mnt/homeGPU/haoweihu/code/models/'+name+'.pkl')
    learn.remove_cb(EarlyStoppingCallback)

    # Validar con el conjunto de test
    loss, acc, recall, f1 = learn.validate(dl=test_dls.train)
    print(f"Loss: {loss}")
    print(f"Accuracy: {acc}")
    print(f"Recall (macro): {recall}")
    print(f"F1 Score (macro): {f1}")

if __name__ == '__main__':
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example1", "cifar10_dist1_quand2_ex1")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example2", "cifar10_dist1_quand2_ex2")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example3", "cifar10_dist1_quand2_ex3")
    
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example1", "cifar10_dist5_quand2_ex1")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example2", "cifar10_dist5_quand2_ex2")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example3", "cifar10_dist5_quand2_ex3")
    
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example1", "cifar10_dist01_quand2_ex1")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example2", "cifar10_dist01_quand2_ex2")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example3", "cifar10_dist01_quand2_ex3")
