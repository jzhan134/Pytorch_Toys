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
train_dl = DataLoader(train_data, batch_size=400, shuffle=True)
val_dl = DataLoader(val_data,batch_size=len(val_data))
#%%
import torch.nn as nn
def conv_block(in_, out_):
    return nn.Sequential(
        nn.Conv2d(in_, out_, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_),
    )

class Demo_Model(nn.Module):
    def __init__(self, in_, out_) -> None:
        super().__init__()
        self.in_ = in_
        self.out_ = out_

        self.conv1 = conv_block(self.in_, 64)
        self.conv2 = conv_block(64, 128)
        self.max_pool1 = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x) # 64*32*32
        out = self.conv2(out) # 128*32*32
        res1 = self.max_pool1(out) # 128*16*16
        out = conv_block(128, 128)(res1)
        out = conv_block(128, 128)(out)
        out = out + res1 #! res1 is the residual
        out = nn.MaxPool2d(4)(out) # 128*4*4
        out = nn.Flatten()(out) # 2048
        out = nn.Dropout(0.2)(out)  #! zero input tensor elements
        out = nn.Linear(2048, self.out_)(out) # 10
        return out


model = Demo_Model(3, 10)
#%%
optimizer_fn = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(20):
    t0 = time.time()
    model.train()
    for img, label in tqdm(train_dl):
        # img, label = img.to(dev), label.to(dev)
        optimizer_fn.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer_fn.step()
    model.eval()
    for img, label in val_dl:
        # img, label = img.to(dev), label.to(dev)
        pred = model(img)
        loss = loss_fn(pred, label)
        pred = torch.argmax(pred, dim=1)
        acc = torch.sum(torch.eq(pred, label)) / len(label)
    print(f"Epoch {epoch}: loss = {loss}, accuracy = {acc.item()}, time = {time.time() - t0:.2f}")
    if acc > best_acc:
        best_acc = acc
        best_model = copy.deepcopy(model)