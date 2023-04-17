import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights2(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

            
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        return i
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input


class SimpleRegressor(nn.Module):
    def __init__(self, num_input, num_output, num_layers=2, hidden_size=100):
        super(SimpleRegressor, self).__init__()
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_input, hidden_size))
            layers.append(nn.ReLU())
            num_input = hidden_size
        layers.append(nn.Linear(num_input, num_output))
        self.network = nn.Sequential(*layers)
        self.apply(init_weights2)

    def forward(self, emb):
        return self.network(emb)