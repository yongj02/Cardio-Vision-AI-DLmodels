from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from sklearn.metrics import roc_auc_score
from metrics import precision, recall, f1
from splitting_data import splitting_data

n = 1
def MLP(num_of_iter, dataframe, target_col, test_size=0.8, random_state=42):
  X = dataframe.drop(columns=[target_col], axis=1)
  y = dataframe[target_col]

  X_train, y_train, X_test, y_test, X_val, y_val = splitting_data(X, y, test_size, random_state)
  n_features = X_train.shape[1]  # Number of features in the dataset

  mlp_loss = mlp_accuracy = mlp_precision = mlp_recall = mlp_f1 = mlp_roc_auc = 0

  for _ in range(num_of_iter):
    mlp_model = Sequential()

    # Input layer and first hidden layer
    mlp_model.add(Dense(64, activation='relu', input_shape=(n_features,)))
    mlp_model.add(Dropout(0.1))  # Add dropout to reduce overfitting

    # Second hidden layers
    mlp_model.add(Dense(32, activation='relu'))
    mlp_model.add(Dropout(0.1))

    # Output layer
    mlp_model.add(Dense(1, activation='sigmoid'))

    # Compiling MLP model
    mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1])

    # Training MLP model
    mlp_model_history = mlp_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

    # Getting results
    curr_loss, curr_accuracy, curr_precision, curr_recall, curr_f1 = mlp_model.evaluate(X_test, y_test, verbose=0)
    mlp_loss += curr_loss
    mlp_accuracy += curr_accuracy
    mlp_precision += curr_precision
    mlp_recall += curr_recall
    mlp_f1 += curr_f1
    mlp_roc_auc += roc_auc_score(y_test, mlp_model.predict(X_test, verbose=0))

  mlp_loss /= num_of_iter
  mlp_accuracy /= num_of_iter
  mlp_precision /= num_of_iter
  mlp_recall /= num_of_iter
  mlp_f1 /= num_of_iter
  mlp_roc_auc /= num_of_iter

  return mlp_loss, mlp_accuracy, mlp_precision, mlp_recall, mlp_f1, mlp_roc_auc