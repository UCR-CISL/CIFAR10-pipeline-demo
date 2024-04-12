import torch
from torchvision import datasets, transforms

def get_cifar10_loader(batch_size, train=True, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_dataset = datasets.CIFAR10(root='./data', train=train, transform=transform, download=download)
    cifar10_loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=train)
    return cifar10_loader

def get_svhn_loader(batch_size, split='train', download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    svhn_dataset = datasets.SVHN(root='./data', split=split, transform=transform, download=download)
    svhn_loader = torch.utils.data.DataLoader(svhn_dataset, batch_size=batch_size, shuffle=split == 'train')
    return svhn_loader
