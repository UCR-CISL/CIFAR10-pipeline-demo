import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loader(batch_size, train=True, download=True, data_root='./data'):
    """
    Create a DataLoader for the CIFAR-10 dataset.
    
    Args:
        batch_size (int): Number of images per batch.
        train (bool): Load training data if True, else load test data.
        download (bool): Download the data if not available at data_root.
        data_root (str): Root directory where data is to be stored.
    
    Returns:
        DataLoader: DataLoader for CIFAR-10 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_dataset = datasets.CIFAR10(root=data_root, train=train, transform=transform, download=download)
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=train)
    return cifar10_loader

def get_svhn_loader(batch_size, split='train', download=True, data_root='./data'):
    """
    Create a DataLoader for the SVHN dataset.
    
    Args:
        batch_size (int): Number of images per batch.
        split (str): 'train' for training set, 'test' for test set, 'extra' for extra set.
        download (bool): Download the data if not available at data_root.
        data_root (str): Root directory where data is to be stored.
    
    Returns:
        DataLoader: DataLoader for SVHN dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    svhn_dataset = datasets.SVHN(root=data_root, split=split, transform=transform, download=download)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=split == 'train')
    return svhn_loader

def streaming_data_loader(data_loader):
    """
    Generator that yields data one batch at a time from a DataLoader.
    
    Args:
        data_loader (DataLoader): DataLoader from which to stream the data.
    
    Yields:
        Tuple[Tensor, Tensor]: A batch of data and labels.
    """
    for data, target in data_loader:
        yield data, target

# Example usage for traditional and streaming data loading
if __name__ == "__main__":
    batch_size = 32

    # Traditional batch loading
    cifar10_loader = get_cifar10_loader(batch_size=batch_size)
    svhn_loader = get_svhn_loader(batch_size=batch_size)

    # Streaming loading (simulated)
    cifar10_stream = streaming_data_loader(cifar10_loader)
    svhn_stream = streaming_data_loader(svhn_loader)

    # Process a few batches to demonstrate streaming
    for i, (data, labels) in enumerate(cifar10_stream):
        print("CIFAR10 Streaming Batch", i, "Data shape:", data.shape, "Labels shape:", labels.shape)
        if i == 2:  # Example limit for streaming demonstration
            break

    for i, (data, labels) in enumerate(svhn_stream):
        print("SVHN Streaming Batch", i, "Data shape:", data.shape, "Labels shape:", labels.shape)
        if i == 2:  # Example limit for streaming demonstration
            break
