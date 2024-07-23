from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .normalisation_encoding import process_numerical_data
import numpy as np
import pandas as pd


imputers = {
    'KNN': KNNImputer(n_neighbors=5),
    'RF': IterativeImputer(estimator=RandomForestRegressor(), random_state=42),
    'MICE': IterativeImputer(random_state=42),
    'CART': IterativeImputer(estimator=DecisionTreeRegressor(), random_state=42),
}


def label_encode_with_nan(dataframe):
  label_encoders = {}
  for column in dataframe.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    # Encode only non-null values
    non_null_values = dataframe[column].dropna().unique()
    le.fit(non_null_values)
    label_encoders[column] = le
    dataframe[column] = dataframe[column].apply(lambda x: le.transform([x])[0] if pd.notnull(x) else x)
  return label_encoders

def label_decoders(dataframe, encoders):
  for key, value in encoders.items():
    mapping_encoders = {i: value for i, value in enumerate(value.classes_)}
    print(mapping_encoders)
    dataframe[key] = dataframe[key].replace(mapping_encoders)

  return dataframe

def imputing_data(dataframe, imputer):
  columns = dataframe.columns
  imputed_data = imputer.fit_transform(dataframe)
  imputed_df = pd.DataFrame(imputed_data)
  imputed_df.columns = columns
  return imputed_df

def introduce_missing_values(dataframe, missing_fraction=0.1):
  df_missing = dataframe.copy()
  missing_info = {}

  for col in df_missing.columns:
    missing_indices = np.random.choice(df_missing.index, size=int(missing_fraction * len(df_missing)), replace=False)
    missing_info[col] = {idx: df_missing.at[idx, col] for idx in missing_indices}
    df_missing.loc[missing_indices, col] = np.nan

  return df_missing, missing_info

def calculate_metrics(imputed_data, missing_info):
  rmse = mae = n = 0

  for col, missing_values in missing_info.items():
    n += 1
    y_true = list(missing_values.values())
    imputed_data_array = imputed_data[str(col)].to_list()
    y_pred = [imputed_data_array[i] for i in list(missing_values.keys())]

    rmse += np.sqrt(mean_squared_error(y_true, y_pred))
    mae += mean_absolute_error(y_true, y_pred)

  rmse /= n
  mae /= n

  return rmse, mae

def imputation(df, imputer='MICE'):
  if imputer not in imputers:
    raise KeyError(f"{imputer} not in imputers")
  
  # Encoding NaN
  encoders = label_encode_with_nan(df)
  df_clean = df.copy().dropna()
  df_clean.reset_index(drop=True, inplace=True)
  df_clean = process_numerical_data(df_clean, standardise=False)

  imputed_df = imputing_data(df, imputers[imputer])
  df = label_decoders(imputed_df, encoders)

  return df


def testing_imputation_techniques(df):
  # Encoding NaN
  encoders = label_encode_with_nan(df)
  df_clean = df.copy().dropna()
  df_clean.reset_index(drop=True, inplace=True)
  df_clean = process_numerical_data(df_clean, standardise=False)

  # Introducing missing values to train
  df_missing, missing_info = introduce_missing_values(df_clean)

  df_knn_imputed = pd.DataFrame(imputing_data(df_missing, imputers['KNN']), columns=df_missing.columns)
  df_rf_imputed = pd.DataFrame(imputing_data(df_missing, imputers['RF']), columns=df_missing.columns)
  df_mice_imputed = pd.DataFrame(imputing_data(df_missing, imputers['MICE']), columns=df_missing.columns)
  df_cart_imputed = pd.DataFrame(imputing_data(df_missing, imputers['CART']), columns=df_missing.columns)

  # Calculating metrics for each imputation techniques
  knn_rmse, knn_mae = calculate_metrics(df_knn_imputed, missing_info)
  rf_rmse, rf_mae = calculate_metrics(df_rf_imputed, missing_info)
  mice_rmse, mice_mae = calculate_metrics(df_mice_imputed, missing_info)
  cart_rmse, cart_mae = calculate_metrics(df_cart_imputed, missing_info)

  print(f"KNN Imputation Metrics: RMSE: {knn_rmse}, MAE: {knn_mae}")
  print(f"Random Forest Imputation Metrics: RMSE: {rf_rmse}, MAE: {rf_mae}")
  print(f"MICE Imputation Metrics: RMSE: {mice_rmse}, MAE: {mice_mae}")
  print(f"CART Imputation Metrics: RMSE: {cart_rmse}, MAE: {cart_mae}")