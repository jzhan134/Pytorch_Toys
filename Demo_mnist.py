'''
Learn MNIST library with linear regression model
Inputs:
    28*28 images reshaped into 1*724 array
Output:
    10-element array, where each is the probably that 
    the image is predicted as a digit 
Model:
    724 to 10 linear regression model with softmax
    train data are used to update the model
    validation data are used to evaluate the model (and find optimal)
    test data are used to test the optimal model
'''
#%%
import torch
import numpy as np
from torchvision.datasets import MNIST 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# ! Customize transform with paramter(s).
class reshape_tensor():
    def __init__(self, x) -> None:
        self.x = x
    def __call__(self, img):
        return img.reshape(-1, self.x)

# ! multiple transforms (first to tensor, then reshape)
dataset = MNIST(
    root='MNIST/', download=True, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        reshape_tensor(28*28)
    ]),
    target_transform=None
)


from torch.utils.data import random_split
train_ds, val_ds = random_split(dataset, [50000, 10000])


# ! Read Unit_Dataset_and_DataLoader.py
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds,batch_size=128,shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)


#%% 
# ! Method 1 custom linear regression model
class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=28*28, # 1D image
            out_features=10 # 10 classifier
        )
        
    def forward(self, xb): #! This is the model(data) method
        return self.linear(xb.reshape(-1,28*28))

model = MnistModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    model.train() # ! set to train mode
    for images, labels in train_loader:
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval() # ! validation phase
    with torch.no_grad(): # ! turn off gradient
        batch_loss, batch_accuracy = [], []
        for images, labels in val_loader:
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            probabilities = torch.nn.functional.softmax(outputs, dim=1) # softmax probablity
            predictions = torch.argmax(probabilities, dim=1)
            accuracy = torch.sum(torch.eq(predictions,labels))/len(predictions)
            batch_loss.append(loss.item())
            batch_accuracy.append(accuracy)
        
        print(f"Epoch {epoch}: Loss = {np.mean(batch_loss)}, Accuracy = {np.mean(batch_accuracy)}")


#%% method 2 CNN
# ! create a custom layer to reshape data (28, 28) to (1, 28*28)
# (just to show the effect. transforms already handles this.)
class reshape_module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x.view(-1,28*28)

model = torch.nn.Sequential(
    reshape_module(),
    # layer1.weight.t() + layer1.bias
    torch.nn.Linear(in_features=28*28, out_features=32),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=32, out_features=10),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    accuracy = 0
    for images, labels in val_loader:
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1) # softmax probablity
        predictions = torch.argmax(probabilities, dim=1)
        accuracy += torch.sum(torch.eq(predictions,labels)).item()
    print(f"Epoch {epoch} accuracy = {accuracy/len(val_loader.dataset)}")


#%%
test_dataset = MNIST(root='data/', train=False,transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset)
acc = 0
for img, label in test_loader:
    outputs = model(img.reshape(-1,28*28))
    prob = torch.argmax(outputs).item()
    if prob == label.item():
        acc += 1
print(acc/len(test_loader))
