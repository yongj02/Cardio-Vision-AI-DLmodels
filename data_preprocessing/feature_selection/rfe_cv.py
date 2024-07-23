from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import os


def rfecv_feature_selection(dataframe, target_col, print_fs):
  X = dataframe.drop(columns=[target_col], axis=1)
  y = dataframe[target_col]

  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

  rfe_cv = RFECV(estimator=RandomForestClassifier(random_state=42), cv=cv, step=1, min_features_to_select=int(os.getenv('min_features')))
  rfe_cv = rfe_cv.fit(X, y)

  # Print the optimal number of features
  if print_fs:
    print("Optimal number of features: %d" % rfe_cv.n_features_)

  # Print the selected features
  if print_fs:
    print("Selected features: %s" % rfe_cv.support_)

  return rfe_cv.get_feature_names_out()