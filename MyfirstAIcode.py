print(2**10)
import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def transform_to_unit_vectors(array):
    # Create an identity matrix of size 10x10
    identity_matrix = np.eye(10)
    # Use the input array to index into the identity matrix
    # Each element in the input array will select the corresponding row from the identity matrix
    unit_vectors = identity_matrix[array]
    
    return unit_vectors
#Define the Dataset first
batch1=unpickle("data_batch_1")
#print(A[b'num_cases_per_batch'])
print(batch1.keys())
trainingdata=np.array(batch1[b'data'])
traininglabel=np.array(batch1[b'labels'])


#labels=transform_to_unit_vectors(batch1[b'labels'])
#now I Just need to combine the input data and labels together. to form a Dataset 
#First try to use the labels as number instead of column vector
trainingdata=torch.tensor(trainingdata,dtype=torch.float64)
traininglabel=torch.tensor(traininglabel,dtype=torch.float64)
dataset = TensorDataset(trainingdata, traininglabel)
dataloader = DataLoader(dataset, batch_size=10000, shuffle=True)




#figureout the optimization!
#NN structure
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#training options
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
