#!/bin/bash

#SBATCH --job-name MDD                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH -w atenea           # Puedes poner dionisio, u otra de las opciones, mira squeue para ver los que estan libres                 

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/haoweihu/env/

export TFHUB_CACHE_DIR=.

python /mnt/homeGPU/haoweihu/code/quantize3_mnist_lenet.py
python /mnt/homeGPU/haoweihu/code/quantize2_mnist_lenet.py

mail -s "Proceso finalizado" haoweihu926@gmail.com <<< "El proceso ha finalizado"