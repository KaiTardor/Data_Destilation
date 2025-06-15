from fastai.vision.all import *
from pathlib import Path
import time

from utils.lenet import LeNet
from utils.utils import *  
from fastai.callback.progress import ProgressCallback

def main(train_path, test_path, name):
    print(f"---------------------------------------------------------------------------")
    print(f"Ejecutando: {name}")
    print(f"---------------------------------------------------------------------------")

    # Crear el DataBlock para entrenamiento y validación
    dblock = DataBlock(
        #blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='training', valid_name='valid'),
        #splitter=RandomSplitter(valid_pct=0.2, seed=42), 
        get_y=parent_label,
        item_tfms=[otsu_threshold_transform, to_rgb, Resize(224)]
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
    #model = LeNet(num_classes=10)
    learn = vision_learner(dls, resnet18, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, Recall(average='macro'), F1Score(average='macro')], pretrained=False, cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=5)])

    # Crear el objeto Learner
    #learn = Learner(
    #    dls, 
    #    model, 
    #    loss_func=CrossEntropyLossFlat(), 
    #    metrics=[accuracy, Recall(average='macro'), F1Score(average='macro')], 
    #    cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=5)]
    #)

    learn.remove_cb(ProgressCallback)

    # Entrenamiento del modelo con fit_one_cycle
    start_time = time.time()
    #learn.fit_one_cycle(30)
    learn.fine_tune(30)
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
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=IndexSplitter([]),
        item_tfms=[otsu_threshold_transform, to_rgb, Resize(224)]
    )

    #test_dls = test_block.dataloaders("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing")
    #test_dls = test_block.dataloaders("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test")
    #test_dls = test_block.dataloaders("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test")
    test_dls = test_block.dataloaders(test_path)

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
    #print(f"---------------------------------------------------------------------------")
    #print("--------------------------MODELOS mnist -----------------------------")
    #print(f"---------------------------------------------------------------------------")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/mnist/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist1_ex1")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/mnist/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist1_ex2")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/mnist/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist1_ex3")

    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/mnist/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist5_ex1")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/mnist/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist5_ex2")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/mnist/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist5_ex3")
    
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/mnist/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist01_ex1")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/mnist/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist01_ex2")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/mnist/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_dist01_ex3")
    
    print(f"---------------------------------------------------------------------------")
    print("--------------------------MODELOS fmnist -----------------------------")
    print(f"---------------------------------------------------------------------------")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/fmnist/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist1_otsu_ex1")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/fmnist/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist1_otsu_ex2")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/fmnist/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist1_otsu_ex3")

    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/fmnist/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist5_orsu_ex1")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/fmnist/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist5_otsu_ex2")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/fmnist/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist5_otsu_ex3")
    
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/fmnist/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist01_otsu_ex1")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/fmnist/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist01_otsu_ex2")
    main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/fmnist/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_dist01_otsu_ex3")
    
    #print(f"---------------------------------------------------------------------------")
    #print("--------------------------MODELOS cifar -----------------------------")
    #print(f"---------------------------------------------------------------------------")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist1_quant3_ex1")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist1_quant3_ex2")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_001/cifar10/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist1_quant3_ex3")

    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist5_quant3_ex1")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist5_quant3_ex2")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_005/cifar10/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist5_quant3_ex3")
    
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example1/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist01_quant3_ex1")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example2/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist01_quant3_ex2")
    #main("/mnt/homeGPU/haoweihu/code/dataset/distilled_0001/cifar10/example3/", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar10_dist01_quant3_ex3")
    
    
    print(f"---------------------------------------------------------------------------")
    print("--------------------------MODELOS BASE-----------------------------")
    print(f"---------------------------------------------------------------------------")
    #main("/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/training", "/mnt/homeGPU/haoweihu/code/dataset/original/mnist_png/testing", "r_mnist_base")
    #main("/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/train", "/mnt/homeGPU/haoweihu/code/dataset/original/fashion_mnist/test", "r_fmnist_base")
    #main("/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/train", "/mnt/homeGPU/haoweihu/code/dataset/original/cifar10/test", "r_cifar_base")