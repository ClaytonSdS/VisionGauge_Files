import torch

# Cosine Annealing Scheduler
def CosineAnnealingLR(optimizer, epochs, learning_rate_min=0):
       return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate_min)

# Warm-up + Cosine Annealing Scheduler
def WarmUpCosineAnnealingLR(optimizer, epochs, warmup_percentage=0.1, learning_rate_min=1e-8):
    warmup_epochs = max(2, int(warmup_percentage * epochs))

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate_min)

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    
    return scheduler
