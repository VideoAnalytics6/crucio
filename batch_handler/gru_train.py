import torch
import torch.optim as optim
from torch.autograd.function import InplaceFunction
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, SubsetRandomSampler

from autoencoder.dataset import (VIDEO_DIR, VideoDataset,
                                 preprocess_video_dataset)
from autoencoder.loss import loss_function
from batch_handler.gru_filter import (GRU_PATH, diversity_regularization,
                                      get_filter, length_regularization,
                                      representativeness_loss,
                                      scores_to_selects)

# Hyperparameter
sampler_rate = 0.02
batch_size = 2
num_epochs = 10
learning_rate = 0.00025
rep_weight = 0.5

# Preprocess dataset
# preprocess_video_dataset(VIDEO_DIR)

# Load dataset
train_dataset = VideoDataset(VIDEO_DIR)
n_train = int(sampler_rate * len(train_dataset))
print(f'Sampled training set contains {n_train} videos')
train_sampler = SubsetRandomSampler(range(n_train))
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler)
print(f'Length of training set (i.e. number of batch) is {len(train_loader)}')

# Define model and optimizer
extractor, gru = get_filter(is_load=True)
# extractor, gru = get_filter()
optimizer = optim.Adam(gru.parameters(), lr=learning_rate)

# Define loss function
criterion = loss_function(3)
# criterion = loss_function(4)
# criterion = loss_function(5)
# criterion = loss_function(6)


class dff_round(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        result = torch.round(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result


# Training model
for epoch in range(num_epochs):
    train_loss = 0
    for videos in train_loader:
        optimizer.zero_grad()

        # Forward propagation
        features, size = extractor(videos)
        scores = gru(features)
        scores = scores_to_selects(scores, True)

        # Set threshold for binary values
        selects = dff_round.apply(scores)
        # Sample probability for binary values
        # selects = torch.bernoulli(Bernoulli(scores).probs)
        # If no frame is selected, first frame is retained by default
        for _ in range(selects.shape[0]):
            if torch.all(selects[_] == 0):
                selects[_][0] = 1

        # Calculate diversity regression
        div_loss = diversity_regularization(features, selects)
        # Calculate representativeness loss
        rep_loss = representativeness_loss(criterion, videos, selects)
        # Calculate weighted loss function
        loss = (1-rep_weight) * div_loss + rep_weight * rep_loss

        # Error propagation and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gru.parameters(), 5.0)
        optimizer.step()
        train_loss += loss.item()

    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

torch.save(gru.state_dict(), GRU_PATH)
