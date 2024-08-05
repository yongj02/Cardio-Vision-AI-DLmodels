'''
from splitting_data import splitting_data
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import numpy as np


def ensemble(dataframe, target_col):
  X = dataframe.drop(columns=[target_col], axis=1)
  y = dataframe[target_col]

  X_train, y_train, X_test, y_test = splitting_data(X, y, train_size=0.8, random_state=42, require_val=False)

  # number of features in the dataset
  n_features = X_train.shape[1]

  # Reshaping the data
  X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
  X_train_reshaped = np.array([np.array(sample).reshape(-1, 1) for sample in X_train_reshaped])

  X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
  X_test_reshaped = np.array([np.array(sample).reshape(-1, 1) for sample in X_test_reshaped])

  # DNN Model
  def create_dnn_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

  dnn_model = create_dnn_model(n_features)

  # CNN Model
  def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

  cnn_model = create_cnn_model((n_features, 1))

  # RNN Model
  def create_rnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

  rnn_model = create_rnn_model((n_features, 1))

  # BiRNN Model
  def create_birnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

  birnn_model = create_birnn_model((n_features, 1))

  # Assuming X_train and y_train are your training data and labels
  batch_size = 32
  epochs = 100

  # Training DNN Model
  dnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

  # Training CNN Model
  cnn_model.fit(X_train_reshaped, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

  # Training RNN Model
  rnn_model.fit(X_train_reshaped, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

  # Training BiRNN Model
  birnn_model.fit(X_train_reshaped, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

  # Build the ensemble model
  def create_ensemble_predictions(models, X_test):
    # Get predictions from each model
    predictions = [model.predict(X_test) for model in models]

    # Stack predictions and compute the average
    stacked_predictions = np.stack(predictions, axis=-1)
    ensemble_predictions = np.mean(stacked_predictions, axis=-1)

    return ensemble_predictions

  # List of trained models
  models = [dnn_model, cnn_model, rnn_model, birnn_model]

  # Assuming X_test is your test data
  ensemble_predictions = create_ensemble_predictions(models, X_test_reshaped)

  ensemble_predictions_binary = (ensemble_predictions > 0.5).astype(int)

  # Calculate accuracy
  ensemble_accuracy = accuracy_score(y_test, ensemble_predictions_binary)

  # Calculate Precision
  ensemble_precision = precision_score(y_test, ensemble_predictions_binary)

  # Calculate Recall
  ensemble_recall = recall_score(y_test, ensemble_predictions_binary)

  # Calculate F1 Score
  ensemble_f1 = f1_score(y_test, ensemble_predictions_binary)

  # Calculate AUC-ROC
  ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions)

  # Calculate Loss
  ensemble_loss = log_loss(y_test, ensemble_predictions)

  # Print the evaluation results
  return ensemble_loss, ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1, ensemble_roc_auc
'''