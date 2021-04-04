import torch.optim.lr_scheduler as lr_scheduler


def stepLR(optimizer, cfg_scheduler):
    step_size = cfg_scheduler.STEP_SIZE
    gamma = cfg_scheduler.GAMMA
    return lr_scheduler.StepLR(optimizer, step_size, gamma)

def exponentialLR(optimizer, cfg_scheduler):
    #gamma = 0.95
    gamma = cfg_scheduler.GAMMA
    return lr_scheduler.ExponentialLR(optimizer, gamma)

def cosineannealLR(optimizer, cfg_scheduler):
    T_max = cfg_scheduler.T_MAX
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max)