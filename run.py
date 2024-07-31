import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from deep_learning_models.woclsav2 import whale_optimization_algorithm
from deep_learning_models.cnnlstma import cnnlstma
from deep_learning_models.ensemble import ensemble
import pandas as pd
from data_preprocessing.imputation import imputation
from data_preprocessing.normalisation_encoding import process_numerical_data, encode_non_numerical_data
from data_preprocessing.feature_selection.execute_fs import execute_fs
import keras
import tensorflow as tf
import datetime
import time
import os
import math
import numpy as np


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


content = "\n"
# Ensemble
'''
for i, d in enumerate(dfs):
  keras.utils.set_random_seed(42)  # 1, 15, 42
  tf.config.experimental.enable_op_determinism()
  tf.config.run_functions_eagerly(True)
  tf.data.experimental.enable_debug_mode()

  model_name = models[i] + "Ensemble"
  model = f'Model: {model_name}'
  print(model)
  # print(d.columns)
  results = ensemble(d, target)
  content += "\n" + model + "\n" + str(results)
  i += 1

content += "\n"
'''
# WOCLSA-V2
opts = {
    'N': 10,       # Number of whales
    'T': 3,        # Maximum number of iterations
    'b': 1         # Constant
}

for i, d in enumerate(dfs):
  print("Current time:", datetime.datetime.now())
  keras.utils.set_random_seed(1)  # 1, 15, 42
  tf.config.experimental.enable_op_determinism()
  tf.config.run_functions_eagerly(True)
  tf.data.experimental.enable_debug_mode()

  model_name = models[i] + "WOCLSA"
  model = f"Model: {model_name}"
  print(model)
  # print("Features:", d.columns)
  start_time = time.time()
  results = whale_optimization_algorithm(d, target, opts)
  end_time = time.time()
  duration = int(np.round((end_time - start_time) / 60))
  content = "\n" + model + f" ({duration})" + "\n" + str(results[:-1])
  with open(fs_file, 'a') as f:
    f.write(content)


print("Current time:", datetime.datetime.now())
'''
random_seeds = [0, 1, 42, 1234, 10, 123, 2, 5]
k = 0
for i in random_seeds:
  if k < 2:
    k += 1
    continue
  print("Current time:", datetime.datetime.now())
  print("Current seed:", i)
  keras.utils.set_random_seed(i)
  tf.config.experimental.enable_op_determinism()
  tf.config.run_functions_eagerly(True)
  tf.data.experimental.enable_debug_mode()

  start_time = time.time()
  result = whale_optimization_algorithm(df, target, opts)
  end_time = time.time()
  duration = int(np.round((end_time - start_time) / 60))
  with open("C:/Users/yjche/PycharmProjects/Monash/FIT4701/datasets/woclsa.txt", 'a') as f:
    f.write(f"{i}({duration}): {result}\n")
'''

'''
# params = [[11, 11, 7, 3], [11, 11, 8, 3], [4, 4, 4, 1], [9, 10, 6, 2]]
params = [[11, 10, 5, 1.5]]

for p in params:
  results = cnnlstma(df, target, neuron1=2**p[0], neuron2=2**p[1], batch_size=2**p[2], dropout_rate=p[3]/10)
  print(results)
'''