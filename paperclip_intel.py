import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


import time
from openvino.runtime import Core
class ResNetGaussian(nn.Module):
    def __init__(self):
        super(ResNetGaussian, self).__init__()
        # Load a pretrained ResNet-18 model
        self.base = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        num_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        # Add new fully connected layers for predicting mean and log variance
        self.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Two outputs: mean and log variance
        )
        
    def forward(self, x):
        features = self.base(x)
        x = self.fc(features)
        mean = x[:, 0:1]
        log_var = x[:, 1:2]
        std = torch.exp(0.5 * log_var)  # Calculate standard deviation
        return mean, std, log_var
def gaussian_nll_loss(mean, log_var, target):
    loss = 0.5 * torch.exp(-log_var) * (target - mean)**2 + 0.5 * log_var
    return torch.mean(loss)


# Define a custom Dataset for loading images and their targets from a DataFrame
class PaperclipsDataset(Dataset):
    def __init__(self, csv_file=None, df=None, images_dir=None, transform=None, is_train=True):
        """
        Either csv_file or df must be provided.
        :param csv_file: Path to CSV file.
        :param df: Pre-loaded DataFrame.
        :param images_dir: Directory where images are stored.
        :param transform: torchvision transforms to be applied on the images.
        :param is_train: If True, expects a 'clip_count' column for targets.
        """
        if df is not None:
            self.data = df.reset_index(drop=True)
        elif csv_file is not None:
            self.data = pd.read_csv(csv_file)
        else:
            raise ValueError("Either csv_file or df must be provided.")
            
        self.images_dir = images_dir
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['id']

        image_path = os.path.join(self.images_dir, f"clips-{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            target = row['clip_count']
            # Wrap target in a tensor (regression target as float)
            target = torch.tensor([target], dtype=torch.float32)
            return image, target
        else:
            # For test, return the image and its id
            return image, image_id







torch.onnx.export(model, dummy_input, "simple_model.onnx", input_names=["input"], output_names=["output"])
print("Model exported to ONNX")

# Step 2: Load ONNX model with OpenVINO
core = Core()
model_onnx = core.read_model("simple_model.onnx")

# Step 3: Compile for CPU
compiled_cpu = core.compile_model(model_onnx, "CPU")
infer_cpu = compiled_cpu.create_infer_request()

# Step 4: Compile for GPU (Intel)
compiled_gpu = core.compile_model(model_onnx, "GPU")
infer_gpu = compiled_gpu.create_infer_request()
# Define a model using a pretrained ResNet-18



# Paths to CSV files and image subdirectory
train_path = os.path.join("clips-data-2020", "train.csv")
test_path = os.path.join("clips-data-2020", "test.csv")
images_subdir = os.path.join("clips-data-2020", "clips")
# Load the train and test CSV files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Split train into training and validation sets (90% for training)
train_pct = 0.9
df_train = train_df.sample(frac=train_pct, random_state=0)
df_validate = train_df.drop(df_train.index)

print(f"Training size: {len(df_train)}")
print(f"Validate size: {len(df_validate)}")

# priunt head of train test and val
print(df_train.head())
print(df_validate.head())
print(test_df.head())


# Define image transforms (using a higher resolution for better feature extraction)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create Dataset objects (assuming PaperclipsDataset is defined)
train_dataset = PaperclipsDataset(df=df_train, images_dir=images_subdir, transform=train_transform, is_train=True)
val_dataset = PaperclipsDataset(df=df_validate, images_dir=images_subdir, transform=val_transform, is_train=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Set device, instantiate the model, define loss and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetGaussian().to(device)



# Set device, instantiate the model, define optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetGaussian().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print("device:", device)

def gaussian_nll_loss(mean, log_var, target):
    # Calculate the loss for each instance
    loss = 0.5 * torch.exp(-log_var) * (target - mean)**2 + 0.5 * log_var
    # Return the average loss over the batch
    return torch.mean(loss)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, targets in train_bar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        mean, std, log_var = model(images)
        loss = gaussian_nll_loss(mean, log_var, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        train_bar.set_postfix(loss=loss.item())
    
    train_loss = running_loss / len(train_loader.dataset)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validate]")
    with torch.no_grad():
        for images, targets in val_bar:
            images, targets = images.to(device), targets.to(device)
            mean, std, log_var = model(images)
            loss = gaussian_nll_loss(mean, log_var, targets)
            val_loss += loss.item() * images.size(0)
            val_bar.set_postfix(loss=loss.item())
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Create test dataset (for which we do not have targets)
test_dataset = PaperclipsDataset(csv_file=test_path, images_dir=images_subdir, transform=val_transform, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Run model on test set to generate predictions
torch.onnx.export(
    model,
    dummy_input,
    "resnet_gaussian.onnx",
    input_names=["input"],
    output_names=["mean", "std", "log_var"],
    opset_version=11  # or higher
)
predictions = []
stds = []
image_ids = []
test_bar = tqdm(test_loader, desc="Testing")
with torch.no_grad():
    for images, ids in test_bar:
        images = images.to(device)
        mean, std, log_var = model(images)
        predictions.extend(mean.cpu().numpy())
        stds.extend(std.cpu().numpy())
        image_ids.extend(ids)

# Save predictions to a DataFrame and display the first few rows
test_results = pd.DataFrame({
    'id': image_ids,
    'clip_count': [pred[0] for pred in predictions],
    'std': [s[0] for s in stds]
})
print(test_results.head())
# save model to current directory
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")
# do predictions on all test data, save to csv
test_results.to_csv('test_results.csv', index=False)
print("Test results saved to test_results.csv")


# Set the model to evaluation mode
model.eval()

# Create a list to store a few examples from the validation set
val_examples = []

# Get a few samples from the validation loader (e.g., 10 examples)
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        mean, std, _ = model(images)
        # Move tensors to CPU and iterate through the batch
        for img, target, pred_mean, pred_std in zip(images.cpu(), targets.cpu(), mean.cpu(), std.cpu()):
            val_examples.append((img, target.item(), pred_mean.item(), pred_std.item()))
        # Break after collecting enough examples (adjust number as needed)
        if len(val_examples) >= 10:
            break

# Define number of examples to display (e.g., 10)
num_examples = min(40, len(val_examples))

# Create a subplot grid
fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(15, 6))
axes = axes.flatten()

for ax, (img, true_val, pred_mean, pred_std) in zip(axes, val_examples[:num_examples]):
    # Convert the tensor image to a NumPy array
    # Assuming the image is in [C, H, W] format, convert to [H, W, C]
    img_np = np.transpose(img.numpy(), (1, 2, 0))
    
    # If the images were normalized during preprocessing, you might need to denormalize them here.
    
    ax.imshow(img_np)
    ax.set_title(f"True: {true_val:.2f}\nPred: {pred_mean:.2f}\nStd: {pred_std:.2f}")
    ax.axis('off')

plt.tight_layout()
plt.show()
