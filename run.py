import os
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import Backend
import torch.multiprocessing as mp

# Setting Environment Variables
os.environ['USE_LIBUV'] = '0'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['PYTHONHASHSEED'] = '42'  # Ensure this is set for reproducibility

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress all FutureWarnings

# Importing Modules
from deep_learning_models.woclsa import whale_optimization_algorithm
from data_preprocessing.imputation import imputation
from data_preprocessing.normalisation_encoding import process_numerical_data, encode_non_numerical_data
from data_preprocessing.feature_selection.execute_fs import execute_fs
import pandas as pd
import datetime
import time
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(rank, world_size, model, df, target, fs_file):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=world_size)

    # Setting the seed for each process
    set_seed(42 + rank)

    content = "\n"
    # WOCLSA
    opts = {
      'N': 10,
      'T': 3,
      'b': 1,
      'lb': [4, 4, 4, 1],
      'ub': [11, 11, 9, 5]
    }

    print("Current time:", datetime.datetime.now())
    print(f"Current model: {model}WOCLSA")
    start_time = time.time()
    results = whale_optimization_algorithm(rank, world_size, df, target, opts)
    end_time = time.time()
    duration = int(np.round((end_time - start_time) / 60))

    # Check memory consumption after the algorithm runs
    allocated_memory = torch.cuda.memory_allocated(device=f'cuda:{rank}') / (1024**3)  # in GB
    reserved_memory = torch.cuda.memory_reserved(device=f'cuda:{rank}') / (1024**3)    # in GB
    max_allocated_memory = torch.cuda.max_memory_allocated(device=f'cuda:{rank}') / (1024**3)  # in GB
    max_reserved_memory = torch.cuda.max_memory_reserved(device=f'cuda:{rank}') / (1024**3)    # in GB

    if rank == 0:
        content = f"\nModel: {model}WOCLSA ({duration})\n{str(results[:-1])}"
        content += f"\nMemory usage: Allocated: {allocated_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB"
        content += f"\nMax memory usage: Allocated: {max_allocated_memory:.2f} GB, Reserved: {max_reserved_memory:.2f} GB\n"
        with open(fs_file, 'a') as f:
            f.write(content)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    set_seed(42)

    # Reading datasets
    filename = "balanced_dataset.xlsx"
    fs_file = "./datasets/fs_balanced_dataset.txt"
    if filename == "balanced_dataset.xlsx":
      data = pd.read_excel('./datasets/' + filename)
    else:
      data = pd.read_csv('./datasets/' + filename)
    df = pd.DataFrame(data)


    if filename == 'cleveland.csv':
      target = 'num'
      df[target] = df[target].apply(lambda x: 1 if x > 0 else x)
      df = df.apply(pd.to_numeric, errors='coerce')
    elif filename == 'CVD_FinalData.csv':
      target = 'TenYearCHD'
      df = df.drop(columns=['id'])
    elif filename == "heart.csv":
      target = 'HeartDisease'
    elif filename == "balanced_dataset.xlsx":
      target = 'Label'
      df = df.drop(columns=["Patient ID"])
    elif filename == "darwin_data.csv":
      target = 'class'
      df = df.drop(columns=["ID"])
      df[target] = df[target].map({'P': 1, 'H': 0})
    elif filename == "toxicity_data.csv":
      target = 'Class'
      df[target] = df[target].map({'Toxic': 1, 'NonToxic': 0})
    elif filename == "arcene_data.csv":
      target = 'labels'
    # print(df.head())

    # Imputation
    if df.isnull().values.any():
      df = imputation(df, target)

    # Standardisation/Normalisation and One-Hot Encoding
    df = process_numerical_data(df, target, standardise=False)  # True for standardising numerical data; False for normalising numerical data
    df = encode_non_numerical_data(df, target)

    '''
    # Executing all feature selection techniques
    os.environ['min_features'] = str(math.floor(math.sqrt(len(df.columns) - 1)))
    best_features, models = execute_fs(df, target)

    with open(fs_file, 'a') as f:
      f.write(f"Base features: {len(df.columns) - 1}\n")
    for m, model in enumerate(models):
      string = model[:-1] + f"({len(best_features[m]) - 1}): "
      string += str(best_features[m]) + "\n"
      with open(fs_file, 'a') as f:
        f.write(string)
    '''

    best_features = []
    with open(fs_file, 'r') as file:
      lines = file.readlines()[1:11]
      for line in lines:
        start_idx = line.find('[')
        end_idx = line.find(']')
        if start_idx != -1 and end_idx != -1:
          string_list = line[start_idx:end_idx+1]
          best_features.append(eval(string_list))

    models = ["GA-", "RF-", "RFE-", "RFECV-", "GWO-", "WOA-", "HHO-", "FA-", "CS-", "BA-"]

    '''
    Printing the results
    for f in best_features:
      print(f)
    '''

    models = [""] + models

    # Applying the feature selection to dataframe
    dfs = [df]
    for i in range(len(best_features)):
      dfs.append(df[best_features[i]])

    for i, d in enumerate(dfs):
        world_size = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {world_size}")
        mp.spawn(main, args=(world_size, models[i], d, target, fs_file), nprocs=world_size, join=True)
