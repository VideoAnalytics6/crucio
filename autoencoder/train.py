import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from autoencoder.dataset import (IMAGE_DIR, ImageDataset,
                                 preprocess_image_dataset)
from autoencoder.loss import loss_function
from autoencoder.network2d import decoder_path, encoder_path, get_networks

# Define hyperparameter
sampler_rate = 0.4
batch_size = 8
# sampler_rate = 0.04
# batch_size = 2
num_epochs = 20
learning_rate = 0.00025

# Preprocess dataset
preprocess_image_dataset(IMAGE_DIR)

# Prepare training dataset
train_dataset = ImageDataset(IMAGE_DIR)
n_train = int(sampler_rate * len(train_dataset))
print(f'Sampled training set contains {n_train} images')
train_sampler = SubsetRandomSampler(range(n_train))
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler)
print(
    f'Length of training set (i.e. number of batch) is {len(train_loader)}')

# Define model and optimizer
encoder, decoder = get_networks(is_load=True)
# encoder, decoder = get_networks()
params = [{'params': net.parameters()} for net in [encoder, decoder]]
optimizer = optim.Adam(params, lr=learning_rate)

# Define loss function
criterion = loss_function(3)
# criterion = loss_function(4)
# criterion = loss_function(5)
# criterion = loss_function(6)

# Training model
for epoch in range(num_epochs):
    train_loss = 0
    for imgs in train_loader:
        optimizer.zero_grad()

        # Forward propagation
        codes = encoder(imgs)
        outputs = decoder(codes)
        loss = criterion(outputs, imgs)

        # Error propagation and optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

torch.save(encoder.state_dict(), encoder_path)
torch.save(decoder.state_dict(), decoder_path)
