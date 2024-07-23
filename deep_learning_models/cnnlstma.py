from sklearn.metrics import roc_auc_score
from keras.api.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Conv1D, Activation, MaxPooling1D, LSTM, Flatten, BatchNormalization, Dropout, Dense, Input
from keras.api.optimizers import SGD, schedules
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .metrics import precision, recall, f1
import tensorflow as tf

# @tf.function
def cnnlstma(dataframe, target_col, neuron1=2048, neuron2=1024, batch_size=32, dropout_rate=0.15):
    # Prepare the data
    label_encoder = LabelEncoder().fit(dataframe[target_col])
    labels = label_encoder.transform(dataframe[target_col])
    classes = list(label_encoder.classes_)

    X = dataframe.drop(columns=[target_col], axis=1)
    y = labels

    # Split data into train and test sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=len(classes))
    y_valid = to_categorical(y_valid, num_classes=len(classes))

    # Reshape the input data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))


    # Define the model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(Conv1D(500, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(250, 1))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(neuron1, activation="relu")) # neuron 1
    model.add(Dense(neuron2, activation="relu")) # neuron 2
    model.add(Dense(len(classes), activation="softmax"))

    # Define a learning rate schedule
    initial_learning_rate = 1e-3
    lr_schedule = schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=100000,
      decay_rate=0.96,
      staircase=True)

    # Use the learning rate schedule in the optimizer
    opt = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", precision, recall, f1])
    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_valid, y_valid), verbose=0)

    # Evaluate the model
    cnnlstma_loss, cnnlstma_accuracy, cnnlstma_precision, cnnlstma_recall, cnnlstma_f1, = model.evaluate(X_valid, y_valid, verbose=0)
    cnnlstma_roc_auc = roc_auc_score(y_valid, model.predict(X_valid, verbose=0))

    return cnnlstma_loss, cnnlstma_accuracy, cnnlstma_precision, cnnlstma_recall, cnnlstma_f1, cnnlstma_roc_auc