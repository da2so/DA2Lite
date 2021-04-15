import os
from abc import ABC, abstractmethod

from torchvision.datasets.mnist import MNIST
from torchvision.datasets import  CIFAR10, CIFAR100
import torchvision.transforms as transforms

from DA2Lite.data.aug import normalize, common

class Public_Dataset(ABC):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    @abstractmethod
    def build(self):
        raise NotImplementedError
    
    def normlaize(self, data_aug, img_shape, mean , std):
        img_size = img_shape[1]
        train_trans_list = []
        if data_aug: 
            train_trans_list += common(img_size)
        else:
            train_trans_list += [transforms.Resize(size=img_size)]
        
        train_trans_list += normalize(mean, std)

        self.train_trans = transforms.Compose(train_trans_list)
        self.test_trans =  transforms.Compose(normalize(mean, std))
        

class MNIST_Dataset(Public_Dataset):
    def __init__(self,
                data_aug,
                img_shape,
                data_dir,
                mean=(0.1307,),
                std=(0.3081,)):
        
        data_aug = False
        super().__init__(data_dir)
        self.normlaize(data_aug, img_shape, mean, std)

    def build(self):
        train_dt = MNIST(self.data_dir, transform=self.train_trans, download=True)
        test_dt = MNIST(self.data_dir, train=False, transform=self.test_trans, download=True)
        
        return train_dt, test_dt


class CIFAR10_Dataset(Public_Dataset):
    def __init__(self, 
                data_aug,
                img_shape,
                data_dir,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)):

        super().__init__(data_dir)
        self.normlaize(data_aug, img_shape, mean, std)

    def build(self):
        train_dt = CIFAR10(self.data_dir, transform=self.train_trans, download=True)
        test_dt = CIFAR10(self.data_dir, train=False, transform=self.test_trans, download=True)
        
        return train_dt, test_dt


class CIFAR100_Dataset(Public_Dataset):
    def __init__(self, 
                data_aug,
                img_shape,
                data_dir,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)):

        super().__init__(data_dir)
        self.normlaize(data_aug, img_shape, mean, std)

    def build(self):
        train_dt = CIFAR100(self.data_dir, transform=self.train_trans, download=True)
        test_dt = CIFAR100(self.data_dir, train=False, transform=self.test_trans, download=True)
        
        return train_dt, test_dt
        

def mnist(data_aug, img_shape, data_dir):
    return MNIST_Dataset(data_aug, img_shape, data_dir).build()

def cifar10(data_aug, img_shape, data_dir):
    return CIFAR10_Dataset(data_aug, img_shape, data_dir).build()

def cifar100(data_aug, img_shape, data_dir):
    return CIFAR100_Dataset(data_aug, img_shape, data_dir).build()