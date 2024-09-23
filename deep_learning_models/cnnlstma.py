import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import torch.nn.functional as F

class CNNLSTMA(nn.Module):
    def __init__(self, input_dim, num_classes, neuron1=2048, neuron2=1024, dropout_rate=0.15):
        super(CNNLSTMA, self).__init__()
        self.conv1 = nn.Conv1d(1, 500, 1)
        self.conv2 = nn.Conv1d(500, 250, 1)
        self.lstm = nn.LSTM(250, 512, batch_first=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512 * (input_dim // 4), neuron1)
        self.fc2 = nn.Linear(neuron1, neuron2)
        self.fc3 = nn.Linear(neuron2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x, _ = self.lstm(x.transpose(1, 2))
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def cnnlstma(dataframe, target_col, neuron1=2048, neuron2=1024, batch_size=32, dropout_rate=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(batch_size)

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

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    X_valid = torch.FloatTensor(X_valid).unsqueeze(1).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_valid = torch.LongTensor(y_valid).to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = CNNLSTMA(X_train.shape[2], len(classes), neuron1, neuron2, dropout_rate).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # Training loop
    best_metrics = {'loss': float('inf'), 'roc_auc': 0, 'TP': None, 'FP': None, 'TN': None, 'FN': None}
    for epoch in range(100):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_true_labels = []
        val_predictions = []

        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()

                # Apply softmax to get probabilities for each class
                probs = F.softmax(outputs, dim=1)

                # Append the predicted probabilities to val_probs
                val_probs.extend(probs.cpu().numpy())

                # Get the predicted class (highest probability)
                _, predicted = torch.max(probs, 1)
                val_true_labels.extend(batch_y.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        # Convert val_probs to a NumPy array
        val_probs = np.array(val_probs)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(val_true_labels, val_predictions)

        # Calculate TP, FP, TN, FN per class
        tp_per_class = []
        fp_per_class = []
        tn_per_class = []
        fn_per_class = []

        for i in range(len(classes)):
            tp = conf_matrix[i, i]  # True Positive is the diagonal element
            fp = conf_matrix[:, i].sum() - tp  # False Positive is column sum minus the diagonal element
            fn = conf_matrix[i, :].sum() - tp  # False Negative is row sum minus the diagonal element
            tn = conf_matrix.sum() - (tp + fp + fn)  # True Negative is everything else

            tp_per_class.append(tp)
            fp_per_class.append(fp)
            tn_per_class.append(tn)
            fn_per_class.append(fn)

        # Calculate ROC AUC score (multi-class or binary)
        val_true_labels_bin = label_binarize(val_true_labels, classes=np.arange(len(classes)))

        if len(classes) == 2:
            val_roc_auc = roc_auc_score(val_true_labels_bin, val_probs[:, 1])
        else:
            val_roc_auc = roc_auc_score(val_true_labels_bin, val_probs, multi_class='ovr', average='weighted')

        # Update best metrics if needed
        val_loss /= len(valid_loader)
        if val_loss < best_metrics['loss']:
            best_metrics = {
                'loss': val_loss,
                'roc_auc': val_roc_auc,
                'TP': tp_per_class,
                'FP': fp_per_class,
                'TN': tn_per_class,
                'FN': fn_per_class
            }

    return best_metrics, model