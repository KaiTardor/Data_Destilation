def get_mnist_dataset():
    os.environ["FASTAI_HOME"] = str(Path('/content/dataset'))

    try:
        path = untar_data(URLs.MNIST)
    except Exception as e:
        print(f"Error al descargar el dataset: {e}")
        
    # Definir rutas para la organización
    base_path = Path('./dataset')
    original_path = base_path / "original"

    original_path.mkdir(parents=True, exist_ok=True)

    dest_original = original_path / path.name
    if not dest_original.exists():
        shutil.move(str(path), str(dest_original))
        print(f"Dataset movido a: {dest_original}")
    else:
        print("El dataset ya se encuentra en la carpeta original.")
        
def get_fmnist_dataset(): 
    os.environ["FASTAI_HOME"] = str(Path('/content/dataset'))
    
    base_path = Path('/content/dataset')
    fashion_path = base_path / "fashion_mnist"  # Ruta donde se descargará FashionMNIST

    # 2. Descargar FashionMNIST con torchvision (se obtienen imágenes PIL al usar transform=None)
    train_dataset = datasets.FashionMNIST(root=str(fashion_path), train=True, download=True)
    test_dataset  = datasets.FashionMNIST(root=str(fashion_path), train=False, download=True)

    # 3. Definir la estructura de carpetas destino: /content/dataset/original/fashion_mnist/{train,test}/{clase}/imagen.png
    dest_base   = base_path / "original" / "fashion_mnist"
    train_folder = dest_base / "train"
    test_folder  = dest_base / "test"

    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    # Obtener los nombres de clases del dataset (por ejemplo: 'T-shirt/top', 'Trouser', etc.)
    classes = train_dataset.classes

    for idx, (img, label) in enumerate(train_dataset):
        label_name = classes[label]
        out_dir = train_folder / label_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{idx}.png"
        img.save(out_file)

    # Guardar imágenes de test organizadas en carpetas por clase
    for idx, (img, label) in enumerate(test_dataset):
        label_name = classes[label]
        out_dir = test_folder / label_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{idx}.png"
        img.save(out_file)
        

def get_mnist_dataset():
    os.environ["FASTAI_HOME"] = str(Path('/content/dataset'))

    try:
        path = untar_data(URLs.CIFAR)
    except Exception as e:
        print(f"Error al descargar el dataset: {e}")
        
    # Definir rutas para la organización
    base_path = Path('./dataset')
    original_path = base_path / "original"

    original_path.mkdir(parents=True, exist_ok=True)

    dest_original = original_path / path.name
    if not dest_original.exists():
        shutil.move(str(path), str(dest_original))
        print(f"Dataset movido a: {dest_original}")
    else:
        print("El dataset ya se encuentra en la carpeta original.")