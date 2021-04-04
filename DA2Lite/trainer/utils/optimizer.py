import torch.optim as optim

def adam(model, lr, cfg_optimizer):
    betas = cfg_optimizer.BETAS
    epsilon = cfg_optimizer.EPSILON
    return optim.Adam(params=model.parameters(), lr=lr, betas=betas, epsilon=epsilon)

def sgd(model, lr, cfg_optimizer):
    momentum = cfg_optimizer.MOMENTUM
    weight_decay = cfg_optimizer.WEIGHT_DECAY
    return optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

