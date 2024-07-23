from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.api.utils import to_categorical


def split_transform(dataframe, target_col):
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

    return X_train, y_train, X_valid, y_valid, len(classes)