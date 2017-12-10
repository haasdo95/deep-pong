import torch
from torch import nn

class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU()
        )
        self.output = nn.Linear(10, 3)
    def forward(self, x):
        out = self.fc1.forward(x)
        out = self.fc2.forward(out)
        out = self.fc3.forward(out)
        out = self.fc4.forward(out)
        out = self.fc5.forward(out)
        out = self.output.forward(out)
        return out
