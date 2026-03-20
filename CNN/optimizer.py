import torch

def Adam(model, learning_rate=1e-3, weight_decay=1e-2):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def AdamW(model, learning_rate=1e-3, weight_decay=1e-2):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def SGD(model, learning_rate=1e-3, momentum=0.9, weight_decay=1e-2):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
