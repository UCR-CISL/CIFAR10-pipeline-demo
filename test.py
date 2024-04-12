import torch
import torch.nn.functional as F
from wrn import WideResNet
from dataloader import get_cifar10_loader, get_svhn_loader

def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    logits_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            logits_list.append(outputs.cpu())

    accuracy = 100.0 * total_correct / total_samples
    logits = torch.cat(logits_list, dim=0)
    return accuracy, logits

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = WideResNet(depth=40, widen_factor=2, dropRate=0.3, num_classes=10).to(device)
model = load_checkpoint(model, "/home/rakshithmahishi/Documents/Energy_OOD/cifar10_wrn_s1_energy_ft_epoch_9.pt")

# Load CIFAR-10 and SVHN test data
cifar10_test_loader = get_cifar10_loader(batch_size=128, train=False)
svhn_test_loader = get_svhn_loader(batch_size=128, split='test')

# Evaluate on CIFAR-10
cifar10_accuracy, cifar10_logits = evaluate_model(model, cifar10_test_loader)
print(f"CIFAR-10 Test Accuracy: {cifar10_accuracy}%")

# Evaluate on SVHN
svhn_accuracy, svhn_logits = evaluate_model(model, svhn_test_loader)
print(f"SVHN Test Accuracy: {svhn_accuracy}%")
