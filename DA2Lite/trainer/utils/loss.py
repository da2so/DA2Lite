import torch

def categorical_crossentropy():
    'Crossentropy'
    return torch.nn.CrossEntropyLoss()

def kld():
    'Kullback-Leivler divergence'
    return torch.nn.KLDivLoss()

def mae():
    'Mean Absolute Error'
    return torch.nn.L1Loss()
    
def mse():
    'Mean Square Error'
    return torch.nn.MSELoss()


