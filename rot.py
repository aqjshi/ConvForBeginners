import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --------------------------
# Load Data
# --------------------------
# Load the training and test data from the .mat files.
train_data = scipy.io.loadmat('DigitsDataTrain.mat')
test_data = scipy.io.loadmat('DigitsDataTest.mat')

# Assuming the .mat files have variables 'XTrain', 'anglesTrain', 'XTest', 'anglesTest'
XTrain = train_data['XTrain']  # shape: (height, width, channels, num_samples)
anglesTrain = train_data['anglesTrain'].squeeze()  # shape: (num_samples,)
XTest = test_data['XTest']
anglesTest = test_data['anglesTest'].squeeze()

# MATLAB data is stored as (height, width, channels, samples)
# Convert to (samples, height, width, channels) for consistency, then to PyTorch shape (samples, channels, height, width)
XTrain = np.transpose(XTrain, (3, 0, 1, 2))
XTest = np.transpose(XTest, (3, 0, 1, 2))

# --------------------------
# Display Training Images
# --------------------------
num_images = 49
idx = np.random.choice(XTrain.shape[0], num_images, replace=False)
fig, axes = plt.subplots(7, 7, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(XTrain[idx[i]].squeeze(), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()

# --------------------------
# Partition Data into Training and Validation
# --------------------------
# Set aside 15% of training data for validation
XTrain, XValidation, anglesTrain, anglesValidation = train_test_split(
    XTrain, anglesTrain, test_size=0.15, random_state=42
)

# --------------------------
# Check Data Normalization
# --------------------------
plt.hist(anglesTrain, bins=30)
plt.xlabel("Rotation Angle")
plt.ylabel("Counts")
plt.tight_layout()
plt.show()

# Note: The images are assumed normalized to [0,1]. If not, scale them accordingly.

# --------------------------
# Prepare Data for PyTorch
# --------------------------
# Convert numpy arrays to torch tensors.
# Also, rearrange image dimensions to (batch, channels, height, width)
XTrain = torch.tensor(XTrain, dtype=torch.float32).permute(0, 3, 1, 2)
XValidation = torch.tensor(XValidation, dtype=torch.float32).permute(0, 3, 1, 2)
XTest = torch.tensor(XTest, dtype=torch.float32).permute(0, 3, 1, 2)

anglesTrain = torch.tensor(anglesTrain, dtype=torch.float32).unsqueeze(1)
anglesValidation = torch.tensor(anglesValidation, dtype=torch.float32).unsqueeze(1)
anglesTest = torch.tensor(anglesTest, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets and DataLoaders
batch_size = 128

train_dataset = TensorDataset(XTrain, anglesTrain)
val_dataset = TensorDataset(XValidation, anglesValidation)
test_dataset = TensorDataset(XTest, anglesTest)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# --------------------------
# Define Neural Network Architecture
# --------------------------
class ConvRegressionNet(nn.Module):
    def __init__(self, num_responses=1):
        super(ConvRegressionNet, self).__init__()
        # Input shape: (batch, 1, 28, 28) assuming grayscale images of size 28x28.
        self.features = nn.Sequential(
            # First block: Conv -> BatchNorm -> ReLU -> AvgPool
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Second block
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Third block: Two convolution layers
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # After two pooling layers, the feature map size is 28 -> 14 -> 7 (assuming input 28x28)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, num_responses)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

model = ConvRegressionNet(num_responses=1)
print(model)

# --------------------------
# Specify Training Options
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Learning rate scheduler: Drop LR by 0.1 every 20 epochs.
def adjust_learning_rate(optimizer, epoch, initial_lr=1e-3, drop_factor=0.1, drop_epoch=20):
    lr = initial_lr * (drop_factor ** (epoch // drop_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# --------------------------
# Train Neural Network
# --------------------------
num_epochs = 50

train_losses = []
val_losses = []

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    # Adjust learning rate
    adjust_learning_rate(optimizer, epoch)
    
    print(f"Epoch {epoch}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

# --------------------------
# Test Network
# --------------------------
model.eval()
test_loss = 0.0
all_predictions = []
all_targets = []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * images.size(0)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

test_loss /= len(test_loader.dataset)
print(f"Test MSE Loss: {test_loss:.4f}")

# Compute RMSE
all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)
rmse = np.sqrt(np.mean((all_predictions - all_targets)**2))
print(f"Test RMSE: {rmse:.4f}")

# --------------------------
# Plot Predictions vs Targets
# --------------------------
plt.scatter(all_predictions, all_targets, marker="+")
plt.xlabel("Prediction")
plt.ylabel("Target")
plt.plot([-60, 60], [-60, 60], "r--")
plt.tight_layout()
plt.show()

# --------------------------
# Make Predictions with New Data
# --------------------------
# Make a prediction for the first test image.
model.eval()
with torch.no_grad():
    X_sample = XTest[0:1].to(device)  # shape: (1, channels, height, width)
    Y_pred = model(X_sample).cpu().numpy()
print("Predicted Angle:", Y_pred[0][0])

# Visualize the sample image with the predicted angle.
plt.imshow(X_sample.cpu()[0].permute(1, 2, 0).squeeze(), cmap='gray')
plt.title("Angle: {:.2f}".format(Y_pred[0][0]))
plt.axis('off')
plt.show()
