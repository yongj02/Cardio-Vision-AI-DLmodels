import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from .metrics import precision, recall, f1
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data.distributed import DistributedSampler

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
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool1d(x, 2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool1d(x, 2)
        x, _ = self.lstm(x.transpose(1, 2))
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def cnnlstma(rank, world_size, dataframe, target_col, neuron1=2048, neuron2=1024, batch_size=32, dropout_rate=0.15):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
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
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_valid = torch.FloatTensor(X_valid).unsqueeze(1)
    y_train = torch.LongTensor(y_train)
    y_valid = torch.LongTensor(y_valid)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    # Create DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)

    # Create DataLoaders with DistributedSampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    # Initialize the model
    model = CNNLSTMA(X_train.shape[2], len(classes), neuron1, neuron2, dropout_rate).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # Training loop
    for epoch in range(100):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    total_val_loss = 0
    all_y_pred = []
    all_y_true = []
    outputs_prob = []
    with torch.no_grad():
        for batch_X, batch_y in valid_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_val_loss += loss.item()

            # Get class predictions
            _, predicted = torch.max(outputs, 1)
            all_y_pred.extend(predicted.cpu().numpy())
            all_y_true.extend(batch_y.cpu().numpy())

            # Get probabilities for ROC AUC
            probs = torch.softmax(outputs, dim=1)[:, 1]
            outputs_prob.extend(probs.cpu().numpy())

    all_y_pred = np.array(all_y_pred)
    all_y_true = np.array(all_y_true)
    outputs_prob = np.array(outputs_prob)

    # Loss
    average_val_loss = total_val_loss / len(valid_loader)

    # Accuracy
    accuracy = (all_y_pred == all_y_true).mean()

    # Precision
    precision_score = precision(all_y_true, all_y_pred)

    # Recall
    recall_score = recall(all_y_true, all_y_pred)

    # F1 Score
    f1_score = f1(all_y_true, all_y_pred)

    # ROC_AUC Score
    roc_auc = roc_auc_score(all_y_true, outputs_prob)

    return average_val_loss, accuracy, precision_score, recall_score, f1_score, roc_auc, model
