import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix, accuracy_score
import numpy as np
from splitting_data import splitting_data

# Check if GPU is available
device = torch.device("cpu")

def ensemble(dataframe, target_col):
    label_encoder = LabelEncoder().fit(dataframe[target_col])
    labels = label_encoder.transform(dataframe[target_col])
    classes = list(label_encoder.classes_)

    X = dataframe.drop(columns=[target_col], axis=1)
    y = labels

    if len(classes) == 2:
        loss_function = nn.BCELoss()
        activation_function = nn.Sigmoid()
        output_nodes = 1
    else:
        loss_function = nn.CrossEntropyLoss()
        activation_function = nn.Softmax(dim=1)
        output_nodes = len(classes)

    X_train, y_train, X_test, y_test = splitting_data(X, y, train_size=0.8, random_state=42, require_val=False)

    n_features = X_train.shape[1]

    # Convert to PyTorch tensors and move to device
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # DNN Model
    class DNNModel(nn.Module):
        def __init__(self, input_dim, output_nodes):
            super(DNNModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 100)
            self.fc2 = nn.Linear(100, 64)
            self.fc3 = nn.Linear(64, 128)
            self.fc4 = nn.Linear(128, output_nodes)
            self.dropout = nn.Dropout(0.2)
            self.activation = activation_function

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.fc4(x)
            return self.activation(x)

    # CNN Model
    class CNNModel(nn.Module):
        def __init__(self, input_shape, output_nodes):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * (input_shape // 2), 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, output_nodes)
            self.dropout = nn.Dropout(0.2)
            self.activation = activation_function

        def forward(self, x):
            x = x.unsqueeze(1)  # Add channel dimension
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return self.activation(x)

    # RNN Model
    class RNNModel(nn.Module):
        def __init__(self, input_shape, output_nodes):
            super(RNNModel, self).__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(128, output_nodes)
            self.activation = activation_function

        def forward(self, x):
            x = x.unsqueeze(2)  # Add feature dimension (batch_size, sequence_length, 1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Take the last time step
            x = self.dropout(x)
            x = self.fc(x)
            return self.activation(x)

    # BiRNN Model
    class BiRNNModel(nn.Module):
        def __init__(self, input_shape, output_nodes):
            super(BiRNNModel, self).__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(256, output_nodes)
            self.activation = activation_function

        def forward(self, x):
            x = x.unsqueeze(2)  # Add feature dimension (batch_size, sequence_length, 1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Take the last time step
            x = self.dropout(x)
            x = self.fc(x)
            return self.activation(x)

    def train_model(model, X_train, y_train, batch_size, epochs):
        model.to(device)  # Move model to GPU if available
        optimizer = optim.Adam(model.parameters())
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_function(outputs.squeeze(), batch_y.float() if len(classes) == 2 else batch_y)
                loss.backward()
                optimizer.step()

    batch_size = 32
    epochs = 100

    dnn_model = DNNModel(n_features, output_nodes).to(device)
    cnn_model = CNNModel(n_features, output_nodes).to(device)
    rnn_model = RNNModel(n_features, output_nodes).to(device)
    birnn_model = BiRNNModel(n_features, output_nodes).to(device)

    train_model(dnn_model, X_train_tensor, y_train_tensor, batch_size, epochs)
    train_model(cnn_model, X_train_tensor, y_train_tensor, batch_size, epochs)
    train_model(rnn_model, X_train_tensor, y_train_tensor, batch_size, epochs)
    train_model(birnn_model, X_train_tensor, y_train_tensor, batch_size, epochs)

    def create_ensemble_predictions(models, X_test):
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(X_test)
                predictions.append(pred.cpu().numpy())
        stacked_predictions = np.stack(predictions, axis=-1)
        ensemble_predictions = np.mean(stacked_predictions, axis=-1)
        return ensemble_predictions

    models = [dnn_model, cnn_model, rnn_model, birnn_model]
    ensemble_predictions = create_ensemble_predictions(models, X_test_tensor)

    if len(classes) == 2:
        ensemble_predictions_labels = (ensemble_predictions > 0.5).astype(int)
    else:
        ensemble_predictions_labels = np.argmax(ensemble_predictions, axis=-1)

    ensemble_loss = log_loss(y_test, ensemble_predictions)

    if len(classes) == 2:
        ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions)
    else:
        ensemble_roc_auc = roc_auc_score(y_test, ensemble_predictions, multi_class='ovr')

    y_pred_classes = np.argmax(ensemble_predictions, axis=1)
    y_true_classes = y_test

    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

    tp_per_class = []
    fp_per_class = []
    tn_per_class = []
    fn_per_class = []

    for i in range(len(classes)):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        tn = conf_matrix.sum() - (tp + fp + fn)

        tp_per_class.append(tp)
        fp_per_class.append(fp)
        tn_per_class.append(tn)
        fn_per_class.append(fn)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(accuracy)

    return ensemble_loss, ensemble_roc_auc, tp_per_class, fp_per_class, tn_per_class, fn_per_class