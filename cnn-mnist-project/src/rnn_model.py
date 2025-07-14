import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer - processes sequences
        self.lstm = nn.LSTM(
            input_size=input_size,      # 28 pixels per row
            hidden_size=hidden_size,    # 128 hidden units
            num_layers=num_layers,      # 2 LSTM layers
            batch_first=True,           # (batch, seq, features)
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Classifier
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, 1, 28, 28) -> (batch, 28, 28)
        # Treat as 28 sequences of 28 pixels each
        x = x.squeeze(1)  # Remove channel dimension: (batch, 28, 28)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
        
        # Take the last output (final hidden state)
        out = out[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout and classifier
        out = self.dropout(out)
        out = self.fc(out)
        return out
