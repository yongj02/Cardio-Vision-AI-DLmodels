from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os


def rfe_feature_selection(dataframe, target_col):
  no_features = int(os.getenv('min_features'))
  X = dataframe.drop(columns=[target_col], axis=1)
  y = dataframe[target_col]

  rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=no_features, step=1)
  rfe.fit(X, y)

  return rfe.get_feature_names_out()