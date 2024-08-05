import os
import logging

# Environment Variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHON_INSTALLER_TYPE'] = 'pip'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['USE_LIBUV'] = '0'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress all FutureWarnings

from deep_learning_models.woclsa import whale_optimization_algorithm
from deep_learning_models.cnnlstma import cnnlstma
# from deep_learning_models.ensemble import ensemble
import pandas as pd
from data_preprocessing.imputation import imputation
from data_preprocessing.normalisation_encoding import process_numerical_data, encode_non_numerical_data
from data_preprocessing.feature_selection.execute_fs import execute_fs
import datetime
import time
import math
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

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

    if rank == 0:
        content = f"\nModel: {model}WOCLSA ({duration})\n{str(results[:-1])}"
        with open(fs_file, 'a') as f:
            f.write(content)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    set_seed(42)

    # Reading datasets
    filename = "arcene_data.csv"
    fs_file = "./datasets/fs_arcene_data.txt"
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
