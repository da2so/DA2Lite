import torch.optim as optim

def adam(model, lr, betas=(0.9, 0.999), epsilon=1e-08):
    return optim.Adam(params=model.parameters(), lr=lr, betas=betas, epsilon=epsilon)

def sgd(model, lr, momentum=0.9, weight_decay=5e-4):
    return optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

