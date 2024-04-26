import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from wrn import WideResNet
from dataloader import get_cifar10_loader

def train_model(model, train_loader, val_loader, epochs, learning_rate, device, writer):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        val_loss = evaluate_model(model, val_loader, criterion, device)
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}")

        # Save the model only at the 99th epoch
        if epoch == 199:
            torch.save(model.state_dict(), f'./checkpoints/checkpoint_epoch_{epoch}.pt')

    print("Finished Training")

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(loader)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    model = WideResNet(depth=40, widen_factor=2, dropRate=0.3, num_classes=10).to(device)

    cifar10_train_loader = get_cifar10_loader(batch_size=128, train=True)
    cifar10_val_loader = get_cifar10_loader(batch_size=128, train=False)

    train_model(model, cifar10_train_loader, cifar10_val_loader, epochs=200, learning_rate=0.001, device=device, writer=writer)
    writer.close()

    # Optionally save the final model state
    torch.save(model.state_dict(), './checkpoints/final_model_after_100_epochs.pt')
