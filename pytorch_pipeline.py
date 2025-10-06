import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
#import kagglehub

#path = kagglehub.dataset_download("hojjatk/mnist-dataset")
#print(f"Dataset downloaded to: {path}")

ROOT = "./kagglehub/datasets"
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {DEVICE} device")

# define a simple transform to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# load the MNIST dataset
train_dataset = datasets.MNIST(root=ROOT, train=True, download=False, transform=transform)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=len(train_dataset), shuffle=False
)
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Compute mean and std
mean = images.mean().item()
std = images.std().item()

print(f"MNIST mean: {mean:.4f}")
print(f"MNIST std: {std:.4f}")

# Now, apply normalization using computed mean and std
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

train_dataset_norm = datasets.MNIST(root=ROOT, train=True, download=False, transform=transform_norm)
train_loader_norm = torch.utils.data.DataLoader(train_dataset_norm, batch_size=64, shuffle=True, num_workers=2)

# Split validation set from training set (e.g., 10% for validation)
val_size = int(0.1 * len(train_dataset_norm))
train_size = len(train_dataset_norm) - val_size
train_dataset_norm, val_dataset_norm = torch.utils.data.random_split(train_dataset_norm, (train_size, val_size))
train_loader_norm = torch.utils.data.DataLoader(train_dataset_norm, batch_size=64, shuffle=True)
val_loader_norm = torch.utils.data.DataLoader(val_dataset_norm, batch_size=64, shuffle=False)

# Load normalized test set
test_dataset_norm = datasets.MNIST(root=ROOT, train=False, download=False, transform=transform_norm)
test_loader_norm = torch.utils.data.DataLoader(test_dataset_norm, batch_size=64, shuffle=False)


# Define simple fully connected neural network using nn.Sequential
model = nn.Sequential(
    nn.Flatten(),  # Flatten(28x28 -> 784)
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)  # Logits output
)

# Print model summary
print(model)

import torch.nn as nn

# Define neural network using nn.Module subclass (Model Option B)
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input (batch_size, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)            # Output logits
        return x

# Instantiate the model
model_b = MNISTModel().to(DEVICE)
print(model_b)


# ----- Loss, optimizer, scheduler -----

# Define loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Define optimizer: Adam with lr=1e-3 and weight decay=1e-4
optimizer = optim.Adam(model_b.parameters(), lr=1e-3, weight_decay=1e-4)

# Optional learning rate scheduler (e.g., StepLR or CosineAnnealingLR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Halve LR every 5 epochs


# ----- Training loop skeleton -----

def train_one_epoch(model, train_loader, criterion, optimizer, device=DEVICE):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # Zero gradients after step

        # Track statistics
        running_loss += loss.item() * inputs.size(0)      # average loss * batch size
        _, predicted = outputs.max(1)                     # get predicted class
        total += targets.size(0)                          # total samples
        correct += predicted.eq(targets).sum().item()     # correct predictions

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ----- Validation loop -----

def validate(model, val_loader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


# ----- Training with logging and metrics -----

num_epochs = 10
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0
best_model_state = None

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model_b, train_loader_norm, criterion, optimizer)
    val_loss, val_acc = validate(model_b, val_loader_norm, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model_b.state_dict()

    # Print per-epoch summary
    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


# ----- Test loop -----

def test_model(model, test_loader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): 
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc


test_loss, test_acc = test_model(model_b, test_loader_norm, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


# ----- Save & load best model -----

# Save the model
torch.save(best_model_state, './mnist_model_best.pth')

# Instantiate the model architecture
loaded_model = MNISTModel()

# Load the trained weights
loaded_model.load_state_dict(torch.load('./mnist_model_best.pth'))

# Print out after running the code:
'''
Epoch 1/10 | Train Loss: 0.2468, Train Acc: 0.9263 | Val Loss: 0.1155, Val Acc: 0.9652
Epoch 2/10 | Train Loss: 0.1014, Train Acc: 0.9686 | Val Loss: 0.0872, Val Acc: 0.9733
Epoch 3/10 | Train Loss: 0.0708, Train Acc: 0.9777 | Val Loss: 0.0858, Val Acc: 0.9753
Epoch 4/10 | Train Loss: 0.0565, Train Acc: 0.9818 | Val Loss: 0.0744, Val Acc: 0.9758
Epoch 5/10 | Train Loss: 0.0433, Train Acc: 0.9859 | Val Loss: 0.0726, Val Acc: 0.9787
Epoch 6/10 | Train Loss: 0.0205, Train Acc: 0.9939 | Val Loss: 0.0591, Val Acc: 0.9818
Epoch 7/10 | Train Loss: 0.0134, Train Acc: 0.9962 | Val Loss: 0.0624, Val Acc: 0.9808
Epoch 8/10 | Train Loss: 0.0147, Train Acc: 0.9955 | Val Loss: 0.0740, Val Acc: 0.9792
Epoch 9/10 | Train Loss: 0.0133, Train Acc: 0.9958 | Val Loss: 0.0740, Val Acc: 0.9797
Epoch 10/10 | Train Loss: 0.0119, Train Acc: 0.9966 | Val Loss: 0.0830, Val Acc: 0.9795
Test Loss: 0.0788, Test Accuracy: 0.9784
'''