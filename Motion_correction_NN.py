import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torchmetrics.image import StructuralSimilarityIndexMeasure

import glob
from skimage.io import imread
from skimage.transform import resize
import os
from torch.utils.data import Dataset, DataLoader

import nibabel as nib

# Custom dataset for motion correction
class DatasetMotionCorrection(Dataset):
    def __init__(self, split, path):
        self.split = split  # Dataset split: train/valid/test
        self.path = path  # Base path to the dataset

        # List all clean .nii files for the given split
        self.filenames = glob.glob(self.path + '/' + self.split + '/clean/*.nii', recursive=True)

        self.num_of_imgs = len(self.filenames)  # Number of images

    def __len__(self):
        return self.num_of_imgs

    def __getitem__(self, index):
        filename = self.filenames[index]  # Get file name

        clean = nib.load(filename).get_fdata()  # Load clean image

        # Get corresponding motion-corrupted file
        filename_motion = filename.replace('clean', 'motion')
        motion = nib.load(filename_motion).get_fdata()

        # Rearrange dimensions to be channel-first (C, H, W)
        clean = np.moveaxis(clean.astype(np.float32), 2, 0)
        motion = np.moveaxis(motion.astype(np.float32), 2, 0)

        # Normalize images
        clean = (clean - np.mean(clean)) / np.std(clean)
        motion = (motion - np.mean(motion)) / np.std(motion)

        # Convert to PyTorch tensors
        clean = torch.Tensor(clean)
        motion = torch.Tensor(motion)

        return motion, clean

# Define data loaders for training and validation
loader = DatasetMotionCorrection(split='train_puvodni_2', path='/home/kohutekova/DP_Kohutekova')
trainloader = DataLoader(loader, batch_size=3, shuffle=True)

loader = DatasetMotionCorrection(split='valid_puvodni_2', path='/home/kohutekova/DP_Kohutekova')
validloader = DataLoader(loader, batch_size=3, shuffle=True, drop_last=False)

# 3D Convolutional Layer with Batch Normalization
class unetConv3(nn.Module):
    def __init__(self, in_size, out_size, filter_size=3, stride=1, pad=1, do_batch=1):
        super().__init__()
        self.do_batch = do_batch  # Flag to enable/disable batch normalization

        self.conv = nn.Conv3d(in_size, out_size, filter_size, stride, pad)
        self.bn = nn.BatchNorm3d(out_size, momentum=0.1)

    def forward(self, inputs):
        outputs = self.conv(inputs)  # Apply convolution

        if self.do_batch:
            outputs = self.bn(outputs)  # Apply batch normalization if enabled
        outputs = F.relu(outputs)  # Apply ReLU activation

        return outputs

# 3D Transposed Convolutional Layer
class unetConvT3(nn.Module):
    def __init__(self, in_size, out_size, filter_size=3, stride=2, pad=1, out_pad=1):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_size, out_size, filter_size, stride=stride, padding=pad, output_padding=out_pad)

    def forward(self, inputs):
        outputs = self.conv(inputs)  # Apply transposed convolution
        outputs = F.relu(outputs)  # Apply ReLU activation
        return outputs

# 3D U-Net Upsampling Block
class unetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp3, self).__init__()

        self.up = unetConvT3(in_size, out_size)  # Transposed convolution for upsampling

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)  # Upsample inputs2

        # Match spatial dimensions by padding
        shape1 = list(inputs1.size())
        shape2 = list(inputs2.size())

        pad = (0, 0, shape1[-2] - shape2[-2], 0, shape1[-1] - shape2[-1], 0)
        inputs2 = F.pad(inputs2, pad)

        # Concatenate along the channel dimension
        return torch.cat([inputs1, inputs2], 1)

# 3D U-Net Model
class Unet3D(nn.Module):
    def __init__(self, filters=(np.array([16, 32, 64, 128])).astype(int), in_size=1, out_size=1):
        super().__init__()

        self.in_size = in_size  # Input channels
        self.out_size = out_size  # Output channels
        self.filters = filters  # Number of filters at each level

        # Downsampling path
        self.conv1 = nn.Sequential(
            unetConv3(in_size, filters[0]),
            unetConv3(filters[0], filters[0]),
            unetConv3(filters[0], filters[0])
        )

        self.conv2 = nn.Sequential(
            unetConv3(filters[0], filters[1]),
            unetConv3(filters[1], filters[1]),
            unetConv3(filters[1], filters[1])
        )

        self.conv3 = nn.Sequential(
            unetConv3(filters[1], filters[2]),
            unetConv3(filters[2], filters[2]),
            unetConv3(filters[2], filters[2])
        )

        # Bottleneck
        self.center = nn.Sequential(
            unetConv3(filters[-2], filters[-1]),
            unetConv3(filters[-1], filters[-1])
        )

        # Upsampling path
        self.up_concat3 = unetUp3(filters[3], filters[3])
        self.up_conv3 = nn.Sequential(unetConv3(filters[2] + filters[3], filters[2]), unetConv3(filters[2], filters[2]))

        self.up_concat2 = unetUp3(filters[2], filters[2])
        self.up_conv2 = nn.Sequential(unetConv3(filters[1] + filters[2], filters[1]), unetConv3(filters[1], filters[1]))

        self.up_concat1 = unetUp3(filters[1], filters[1])
        self.up_conv1 = nn.Sequential(unetConv3(filters[0] + filters[1], filters[0]), unetConv3(filters[0], filters[0], do_batch=0))

        # Final 1x1 convolution to produce output
        self.final = nn.Conv3d(filters[0], self.out_size, 1)

        # Initialize weights
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, inputs):
        # Forward pass through downsampling path
        conv1 = self.conv1(inputs)
        x = F.max_pool3d(conv1, 2, 2)

        conv2 = self.conv2(x)
        x = F.max_pool3d(conv2, 2, 2)

        conv3 = self.conv3(x)
        x = F.max_pool3d(conv3, 2, 2)

        # Bottleneck
        x = self.center(x)

        # Forward pass through upsampling path
        x = self.up_concat3(conv3, x)
        x = self.up_conv3(x)

        x = self.up_concat2(conv2, x)
        x = self.up_conv2(x)

        x = self.up_concat1(conv1, x)
        x = self.up_conv1(x)

        # Final output
        x = self.final(x)
        return x

# Clear GPU cache to avoid memory issues
torch.cuda.empty_cache()

# Define training configurations
device = torch.device("cuda:0")  # Use GPU if available
mu = 0.01  # Learning rate
milestones = [125, 250, 375, 450, 500]  # Epoch milestones for learning rate adjustment
epochs = milestones[-1]

best_val_loss = float('inf')  # Track best validation loss

# Mean Squared Error (MSE) Loss function
def MSE(X, Y):
    mse = torch.mean((X - Y) ** 2)
    return mse

criterion = MSE

# SSIM Metric
ssim_metric = StructuralSimilarityIndexMeasure(data_range=None).to(device)
ssim_train = []  # Store SSIM for training
ssim_valid = []  # Store SSIM for validation

# Instantiate the U-Net model
net = Unet3D(in_size=1, out_size=1)
net = net.to(device)

# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=mu)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

loss_train = []  # Track training loss
loss_valid = []  # Track validation loss

# Training loop
for ep in range(epochs):
    print(ep)
    loss_tmp = []
    ssim_tmp_train = []
    net.train()  # Set model to training mode

    for it, (batch, lbls) in enumerate(trainloader):
        batch = batch.to(device)
        lbls = lbls.to(device)

        batch = batch[:, np.newaxis, ...]  # Add channel dimension
        lbls = lbls[:, np.newaxis, ...]

        y_hat = net(batch)  # Forward pass
        L = criterion(y_hat, lbls)  # Calculate loss

        optimizer.zero_grad()  # Zero gradients
        L.backward()  # Backpropagate error
        optimizer.step()  # Update weights

        loss_tmp.append(L.detach().cpu().numpy())  # Store loss

        # Calculate SSIM for batch
        ssim_value = ssim_metric(y_hat, lbls)
        ssim_tmp_train.append(ssim_value.item())

    loss_train.append(np.mean(loss_tmp))
    ssim_train.append(np.mean(ssim_tmp_train))

    # Validation phase
    loss_tmp = []
    ssim_tmp_valid = []
    net.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for it, (batch, lbls) in enumerate(validloader):
            batch = batch.to(device)
            lbls = lbls.to(device)

            batch = batch[:, np.newaxis, ...]  # Add channel dimension
            lbls = lbls[:, np.newaxis, ...]

            y_hat = net(batch)  # Forward pass
            L = criterion(y_hat, lbls)  # Calculate loss

            loss_tmp.append(L.detach().cpu().numpy())  # Store loss

            # Calculate SSIM for batch
            ssim_value = ssim_metric(y_hat, lbls)
            ssim_tmp_valid.append(ssim_value.item())

    loss_valid.append(np.mean(loss_tmp))
    ssim_valid.append(np.mean(ssim_tmp_valid))

    # Save best model
    if loss_valid[-1] < best_val_loss:
        best_val_loss = loss_valid[-1]
        torch.save(net, '/home/kohutekova/DP_Kohutekova/modely/model_6.pth')
        print(f"Best model saved with validation loss {best_val_loss:.4f}")

# Save training and validation metrics to Excel files
df_loss_train = pd.DataFrame(loss_train)
df_loss_train.to_excel('/home/kohutekova/DP_Kohutekova/modely/loss_train_6.xlsx', index=False)
df_loss_valid = pd.DataFrame(loss_valid)
df_loss_valid.to_excel('/home/kohutekova/DP_Kohutekova/modely/loss_valid_6.xlsx', index=False)

df_ssim_train = pd.DataFrame(ssim_train)
df_ssim_train.to_excel('/home/kohutekova/DP_Kohutekova/modely/ssim_train_6.xlsx', index=False)

df_ssim_valid = pd.DataFrame(ssim_valid)
df_ssim_valid.to_excel('/home/kohutekova/DP_Kohutekova/modely/ssim_valid_6.xlsx', index=False)

# Testing phase
output_dir = "/home/kohutekova/DP_Kohutekova/test_results_model6/"
os.makedirs(output_dir, exist_ok=True)

# Prepare test dataset and dataloader
test_loader = DatasetMotionCorrection(split='test_puvodni_2', path='/home/kohutekova/DP_Kohutekova')
test_dataloader = DataLoader(test_loader, batch_size=1, shuffle=False)

# Load the trained model
device = torch.device("cuda:0")
model_path = '/home/kohutekova/DP_Kohutekova/modely/model_6.pth'
model = torch.load(model_path, map_location=torch.device('cuda:0'))
model = model.to(device)
model.eval()  # Set to evaluation mode

# Initialize metrics
test_loss = []
test_ssim = []

# Criterion for testing
def MSE(X, Y):
    mse = torch.mean((X - Y) ** 2)
    return mse

criterion = MSE
ssim_metric = StructuralSimilarityIndexMeasure(data_range=None).to(device)

# Evaluate on test set
with torch.no_grad():
    for batch_idx, (batch, lbls) in enumerate(test_dataloader):
        batch = batch.to(device)
        lbls = lbls.to(device)

        batch = batch[:, np.newaxis, ...]  # Add channel dimension
        lbls = lbls[:, np.newaxis, ...]

        # Make predictions
        predictions = model(batch)

        # Compute loss
        loss = criterion(predictions, lbls)
        test_loss.append(loss.cpu().numpy())

        # Compute SSIM
        ssim_value = ssim_metric(predictions, lbls)
        test_ssim.append(ssim_value.item())

        # Save predictions as NIfTI files
        for i in range(predictions.shape[0]):
            prediction_nifti = nib.Nifti1Image(predictions[i, 0].cpu().numpy(), affine=np.eye(4))
            nib.save(prediction_nifti, os.path.join(output_dir, f"prediction_valid_batch2{batch_idx}_sample{i}.nii"))

# Calculate mean loss and SSIM for the test set
mean_test_loss = np.mean(test_loss)
mean_test_ssim = np.mean(test_ssim)

print(f"Test Loss: {mean_test_loss:.4f}")
print(f"Test SSIM: {mean_test_ssim:.4f}")
