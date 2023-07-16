import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from autoencoder.dataset import (VideoDataset, CNN_FRAME_NUM,
                                 preprocess_video_dataset, VIDEO_DIR)
from autoencoder.loss import loss_function
from autoencoder.network3d import (decoder3d_path, encoder3d_path,
                                   get_networks3d)

# Define hyperparameter
sampler_rate = 0.2
batch_size = 2
num_epochs = 10
learning_rate = 0.00025

# Preprocess dataset
preprocess_video_dataset(VIDEO_DIR)

# Prepare training dataset
train_dataset = VideoDataset(VIDEO_DIR)
n_train = int(sampler_rate * len(train_dataset))
print(f'Sampled training set contains {n_train} videos')
train_sampler = SubsetRandomSampler(range(n_train))
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler)
print(f'Length of training set (i.e. number of batch) is {len(train_loader)}')

# Define model and optimizer
encoder3d, decoder3d = get_networks3d(is_load=True)
# encoder3d, decoder3d = get_networks3d()
params = [{'params': net.parameters()} for net in [encoder3d, decoder3d]]
optimizer = optim.Adam(params, lr=learning_rate)

# Define loss function
criterion = loss_function(3)
# criterion = loss_function(4)
# criterion = loss_function(5)
# criterion = loss_function(6)

# Training model
for epoch in range(num_epochs):
    train_loss = 0
    for videos in train_loader:
        optimizer.zero_grad()

        # Forward propagation
        codes = encoder3d(videos)
        outputs = decoder3d(codes)
        loss = 0
        for frame in range(CNN_FRAME_NUM):
            output_frames = outputs[:, :, frame]
            video_frames = videos[:, :, frame]
            loss += criterion(output_frames, video_frames)
        loss /= CNN_FRAME_NUM

        # Error propagation and optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

torch.save(encoder3d.state_dict(), encoder3d_path)
torch.save(decoder3d.state_dict(), decoder3d_path)
