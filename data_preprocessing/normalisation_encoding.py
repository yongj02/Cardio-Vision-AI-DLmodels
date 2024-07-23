from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import pandas as pd

# Preprocessing numerical data
def process_numerical_data(dataframe, target_col, standardise):
  scaler = StandardScaler() if standardise else MinMaxScaler()
  data = dataframe.drop(columns=[target_col], axis=1)
  numerical_col = data.select_dtypes(include=['number']).columns
  if len(numerical_col) == 0:
      return dataframe
  dataframe[numerical_col] = dataframe[numerical_col].astype('float64')
  numerical_data = dataframe[numerical_col]
  scaler.fit(numerical_data)
  dataframe.loc[:, numerical_col] = scaler.transform(numerical_data)
  return dataframe

# Using OneHotEncoder to encode non-numerical data
def encode_non_numerical_data(dataframe, target_col):
  y = dataframe[target_col]
  dataframe = dataframe.drop(columns=[target_col], axis=1)
  encoder = OneHotEncoder(sparse_output=False)
  non_numerical_col = dataframe.select_dtypes(exclude=['number']).columns
  if len(non_numerical_col) == 0:
    return pd.concat([dataframe, y], axis=1)
  non_numerical_data = dataframe[non_numerical_col]
  encoder.fit(non_numerical_data)
  encoded_data = encoder.transform(non_numerical_data)
  encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(non_numerical_col))

  # Concantenate the encoded data with the original dataframe
  dataframe = pd.concat([dataframe.drop(non_numerical_col, axis=1), encoded_df], axis=1)
  dataframe = pd.concat([dataframe, y], axis=1)
  return dataframe


def normalising_encoding(df):
  # Standardising/Normalising and Encoding data
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = process_numerical_data(df, standardise=False)  # True for standardising numerical data; False for normalising numerical data
    df = encode_non_numerical_data(df)