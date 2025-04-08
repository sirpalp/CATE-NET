
# lstm_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def train_lstm(X, y, input_size, num_classes=3, test_size=0.2, num_epochs=20, batch_size=32, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)

        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(y_true, outputs.cpu().numpy(), multi_class='ovr')

    return model, (acc, prec, rec, auc), y_pred
