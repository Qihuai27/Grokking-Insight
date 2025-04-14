import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Dataset class for image-text pairs
class ImageTextDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
        self.labels = [self._get_label(f) for f in self.text_files]
    
    def _get_label(self, text_file):
        with open(os.path.join(self.text_dir, text_file), 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            label = int(last_line.split(':')[-1].strip())
            return label
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        text_path = os.path.join(self.text_dir, self.text_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

# Define ResNet-18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Data preprocessing transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prepare dataset
image_dir = './shape-4-tan-white/out-shape-img'
text_dir = './shape-4-tan-white/out-shape-txt'
dataset = ImageTextDataset(image_dir, text_dir, transform=transform)

# Split dataset into train and test sets
train_size = 3660
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# Initialize model, loss function, and optimizer
model = ResNet18(num_classes=52).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Initialize lists to store accuracies
train_accuracies = []
test_accuracies = []
epochs = []

# Function to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create output directories
ensure_dir('./resnet18-4-white')
ensure_dir('./resnet18-4-white/model_weights')
ensure_dir('./resnet18-4-white/accuracy_plots')

# Training loop
num_epochs = 20000
for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Print training stats
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'Train Accuracy: {100 * correct_train / total_train:.2f}%')
    train_accuracy = correct_train / total_train

    if epoch % 100 == 0:
        with open('./resnet18-4-white/train_accuracy_file', 'a') as f:
            f.write(f'{epoch+1},{train_accuracy:.2f}\n')

    # Evaluate on test set
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_accuracy = correct / total
            print(f'Test Accuracy: {100 * correct / total:.2f}%')
            with open('./resnet18-4-white/test_accuracy_file', 'a') as f:
                f.write(f'{epoch+1},{test_accuracy:.2f}\n')

        # Save accuracies for plotting
        epochs.append(epoch)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Generate and save plot using matplotlib
        max_epoch = max(epochs)
        max_train_acc = max(train_accuracies)
        max_test_acc = max(test_accuracies)

        plt.figure(figsize=(8, 6))

        # Plot train and test accuracy
        plt.plot(epochs, train_accuracies, color='blue', label='Train Accuracy')
        plt.plot(epochs, test_accuracies, color='green', label='Test Accuracy')

        # Adding titles and labels
        plt.title('Train and Test Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')

        # Save the plot
        plt.savefig('./resnet18-4-white/accuracy_plots/accuracy_plot.png')

        # Save model weights
        torch.save(model.state_dict(), f'./resnet18-4-white/model_weights/{epoch}.pth')
