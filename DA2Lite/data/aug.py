
import torchvision.transforms as transforms


def common(img_size):
    return [transforms.RandomCrop(img_size, padding=img_size//8), transforms.RandomHorizontalFlip()]

def normalize(mean, std):
    return [transforms.ToTensor(), transforms.Normalize(mean, std)]
