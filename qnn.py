import torch
from torch import nn

class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(5, 12),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU()
        )
        self.output = nn.Linear(24, 3)
    def forward(self, x):
        out = self.fc1.forward(x)
        out = self.fc2.forward(out)
        out = self.output.forward(out)
        return out
