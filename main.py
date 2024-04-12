import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde
from wrn import WideResNet
from dataloader import get_cifar10_loader, get_svhn_loader
from test import load_checkpoint

def calculate_energy_scores(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def calculate_fpr95(tpr, fpr):
    fpr95 = np.interp(0.95, tpr, fpr)
    return fpr95

def plot_energy_scores_kde(energy_scores_cifar10, energy_scores_svhn, fpr95):
    scores_range = np.linspace(min(min(energy_scores_cifar10), min(energy_scores_svhn)),
                               max(max(energy_scores_cifar10), max(energy_scores_svhn)), 1000)

    kde_cifar10 = gaussian_kde(energy_scores_cifar10)
    kde_svhn = gaussian_kde(energy_scores_svhn)

    plt.figure(figsize=(10, 6))
    plt.plot(scores_range, kde_cifar10(scores_range), label='CIFAR-10', color='blue')
    plt.plot(scores_range, kde_svhn(scores_range), label='SVHN', color='orange')
    plt.fill_between(scores_range, kde_cifar10(scores_range), alpha=0.5, color='blue')
    plt.fill_between(scores_range, kde_svhn(scores_range), alpha=0.5, color='orange')
    plt.xlabel('Energy Score')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Energy Score Distribution (FPR95: {fpr95:.2f})')
    plt.show()

def evaluate_model(model, data_loader, device):
    model.eval()
    total_logits = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            logits = model(images)
            total_logits.append(logits.detach())
    return torch.cat(total_logits, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model
    checkpoint_path = "cifar10_wrn_s1_energy_ft_epoch_9.pt"
    model = WideResNet(depth=40, widen_factor=2, dropRate=0.3, num_classes=10)
    model = load_checkpoint(model, checkpoint_path)
    model.to(device)

    # Load CIFAR-10 and SVHN test data
    cifar10_test_loader = get_cifar10_loader(batch_size=128, train=False)
    svhn_test_loader = get_svhn_loader(batch_size=128, split='test')

    # Evaluate on CIFAR-10 and SVHN
    cifar10_logits = evaluate_model(model, cifar10_test_loader, device)
    svhn_logits = evaluate_model(model, svhn_test_loader, device)

    # Calculate energy scores
    energy_scores_cifar10 = calculate_energy_scores(cifar10_logits.cpu())
    energy_scores_svhn = calculate_energy_scores(svhn_logits.cpu())

    # Calculate FPR95
    true_labels = np.concatenate([np.zeros(energy_scores_cifar10.shape[0]), np.ones(energy_scores_svhn.shape[0])])
    scores = np.concatenate([energy_scores_cifar10.numpy(), energy_scores_svhn.numpy()])
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    fpr95 = calculate_fpr95(tpr, fpr)

    # Plot energy scores using KDE
    plot_energy_scores_kde(energy_scores_cifar10, energy_scores_svhn, fpr95)

if __name__ == '__main__':
    main()
