import torch 
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
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
batch2=unpickle("data_batch_2")
batch3=unpickle("data_batch_3")
batch4=unpickle("data_batch_4")
batch5=unpickle("data_batch_5")
testdata=unpickle("test_batch")
#print(A[b'num_cases_per_batch'])
#print(batch1.keys())
trainingdata=np.concatenate((np.array(batch1[b'data']),np.array(batch2[b'data']),np.array(batch3[b'data']),np.array(batch4[b'data']),np.array(batch5[b'data'])), axis=0)
traininglabel=np.concatenate((np.array(batch1[b'labels']),np.array(batch2[b'labels']),np.array(batch3[b'labels']),np.array(batch4[b'labels']),np.array(batch5[b'labels'])), axis=0)
testingdata=np.array(testdata[b'data'])
testinglabel=np.array(testdata[b'labels'])



#labels=transform_to_unit_vectors(batch1[b'labels'])
#now I Just need to combine the input data and labels together. to form a Dataset 
#First try to use the labels as number instead of column vector
trainingdata=torch.tensor(trainingdata/256,dtype=torch.float32)
traininglabel=torch.tensor(traininglabel,dtype=torch.float32)
testinglabel=torch.tensor(testinglabel,dtype=torch.float32)
testingdata=torch.tensor(testingdata/256,dtype=torch.float32)
#For CNN
trainingdata=torch.reshape(trainingdata,(50000,3,32,32))
testingdata=torch.reshape(testingdata,(10000,3,32,32))

train_dataset = TensorDataset(trainingdata, traininglabel)
test_dataset=TensorDataset(testingdata,testinglabel)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader=DataLoader(test_dataset, batch_size=16, shuffle=True)
#train_dataloader,test_dataloader=train_dataloader.type(torch.LongTensor),test_dataloader.type(torch.LongTensor)

#figureout the optimization!
#NN structure
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3072, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model1 = NeuralNetwork()   
#SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 20)  # Adjust depending on input size and pooling layers
        self.fc2 = nn.Linear(20, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = SimpleCNN()


#training options

learning_rate = 1e-3
batch_size = 64
epochs = 99
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y = y.long()
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

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            y = y.long()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
##Actuall process
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")