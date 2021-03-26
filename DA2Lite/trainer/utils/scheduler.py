import torch.optim.lr_scheduler as lr_scheduler


def stepLR(optimizer, step_size=60, gamma=0.1):
    return lr_scheduler.StepLR(optimizer, step_size, gamma)


def cosineannealLR(optimizer, T_max=150):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max)