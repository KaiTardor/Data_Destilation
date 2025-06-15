#!/bin/bash

#SBATCH --job-name MDD                 # Nombre del proceso

#SBATCH --partition dios   # Cola para ejecutar

#SBATCH -w titan           # Puedes poner dionisio, u otra de las opciones, mira squeue para ver los que estan libres                 

#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --mem=20GB


export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/haoweihu/TFG/

export TFHUB_CACHE_DIR=.

#python /mnt/homeGPU/haoweihu/quantize/eficientnet_select_random.py
python /mnt/homeGPU/haoweihu/quantize/data_select_eff2.py


mail -s "Proceso finalizado" haoweihu926@gmail.com <<< "El proceso ha finalizado"