#%%
from turtle import forward
import torch
import time
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from torchsummary import summary 

data = torchvision.datasets.CIFAR10(
    root='./CIFAR10', train=True, download=True,
    transform=transforms.ToTensor()
)
train_data, val_data = random_split(data, (45000, len(data) - 45000))
train_dl = DataLoader(train_data, batch_size=128, shuffle=True)
val_dl = DataLoader(val_data,batch_size=100)
#%%
import torch.nn as nn
class Demo_Model(nn.Module):
    def conv_block(self, in_, out_, hasPool=False):
        layers:nn.Sequential = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_),
        )
        if hasPool:
            layers.append(nn.MaxPool2d(2))
        return layers

    def __init__(self, in_, out_) -> None:
        super().__init__()
        self.conv1 = self.conv_block(in_, 64)
        self.conv2 = self.conv_block(64, 128, True)
        self.res1 = nn.Sequential(
            self.conv_block(128, 128), 
            self.conv_block(128, 128)
        )

        self.conv3 = self.conv_block(128, 256, True)
        self.conv4 = self.conv_block(256, 512, True)
        self.res2 = nn.Sequential(
            self.conv_block(512, 512), 
            self.conv_block(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), 
            nn.Dropout(0.2),nn.Linear(512, out_)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

#%%
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Demo_Model(3, 10)
model.to(dev)
optimizer_fn = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(dev)

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

for epoch in range(20):
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
    acc = 0
    for img, label in val_dl:
        img, label = img.to(dev), label.to(dev)
        pred = model(img)
        loss = loss_fn(pred, label)
        pred = torch.argmax(pred, dim=1)
        acc += torch.sum(torch.eq(pred, label)) / len(label)
    print(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {acc/50:.4f}, time = {time.time() - t0:.2f}")
    if acc/50 > best_acc:
        best_acc = acc/50
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