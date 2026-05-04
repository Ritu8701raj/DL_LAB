import torch
import torch.nn as nn

class SimpleGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x
