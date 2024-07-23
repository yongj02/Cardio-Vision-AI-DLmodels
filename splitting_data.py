from sklearn.model_selection import train_test_split

def splitting_data(X, y, train_size=0.8, random_state=42, require_val=True):
  test_size = 1 - train_size
  if require_val:
    test_size /= 2

  X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  if not require_val:
    return X_train_full, y_train_full, X_test, y_test

  X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=random_state)

  return X_train, y_train, X_test, y_test, X_val, y_val