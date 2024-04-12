import torch
import torch.nn as nn
import torch.optim as optim
from wrn import WideResNet
from dataloader import get_cifar10_loader

def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def train_model(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("Finished Training")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = WideResNet(depth=40, widen_factor=2, dropRate=0.3, num_classes=10).to(device)
model = load_checkpoint(model, "/home/rakshithmahishi/Documents/Energy_OOD/cifar10_wrn_s1_energy_ft_epoch_9.pt")

# Load CIFAR-10 training data
cifar10_train_loader = get_cifar10_loader(batch_size=128, train=True)

# Train the model (fine-tuning)
train_model(model, cifar10_train_loader, epochs=5, learning_rate=0.001)
