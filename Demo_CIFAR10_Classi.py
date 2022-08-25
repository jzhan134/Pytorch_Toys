#%%
import torch
import time
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import ssl
from tqdm import tqdm
ssl._create_default_https_context = ssl._create_unverified_context

data = torchvision.datasets.CIFAR10(
    root='./CIFAR10', train=True, download=True,
    transform=transforms.ToTensor()
)
train_data, val_data = random_split(data, (45000, len(data) - 45000))
train_dl = DataLoader(train_data, batch_size=128, shuffle=True)
val_dl = DataLoader(val_data,batch_size=100)

#%%
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(
        in_channels=3, out_channels=32, kernel_size=3,padding=1
    ),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

    nn.Flatten(), 
    nn.Linear(256*4*4, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
optimizer_fn = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# ! Continue the training if possible
import copy
import os
if os.path.exists('CIFAR10_checkpoint.pt'):
    checkpoint = torch.load('CIFAR10_checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer_fn.load_state_dict(checkpoint['opt'])
    best_model = checkpoint['best_model']
    best_acc = checkpoint['best_acc']
    epoch0 = checkpoint['epoch']
    print('Existing model loaded.')
else:
    best_acc = float('-inf')
    epoch0 = 0
    best_model = None
    print('New model created.')
#%%
# dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dev = 'cpu'
model.to(dev)
loss_fn.to(dev)
print(f'learning using {dev}')
for epoch in range(epoch0, 20 + epoch0):
    t0 = time.time()
    model.train()
    for img, label in tqdm(train_dl):
        img, label = img.to(dev), label.to(dev)
        optimizer_fn.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer_fn.step()
    model.eval()
    tot_acc = 0
    for img, label in tqdm(val_dl):
        img, label = img.to(dev), label.to(dev)
        pred = model(img)
        loss = loss_fn(pred, label)
        pred = torch.argmax(pred, dim=1)
        tot_acc += torch.sum(torch.eq(pred, label)) / len(label)
    print(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {tot_acc/50:.4f}, time = {time.time() - t0:.2f}")
    if tot_acc/50 > best_acc:
        best_acc = tot_acc/50
        best_model = copy.deepcopy(model)
#%%
checkpoint = {
    'epoch': epoch + 1,
    'model': model.state_dict(),
    'opt': optimizer_fn.state_dict(),
    'best_model': best_model,
    'best_acc': best_acc
}
torch.save(checkpoint,'CIFAR10_checkpoint.pt')

#%%
imgs = torch.tensor(data.data[:625]).permute((0,3,1,2))
grid = torchvision.utils.make_grid(imgs,nrow=25)
img = transforms.ToPILImage()(grid)
img.show()