#%%
import torch
from torchsummary import summary

model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch', 
    'unet',
    in_channels=3, 
    out_channels=1, 
    init_features=32, 
    pretrained=True
)
summary(model)

#%%
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data = torchvision.datasets.CIFAR10(
    root='./CIFAR10', train=True, download=True,
    transform=transforms.ToTensor()
)
train_dl = DataLoader(data, batch_size=1, shuffle=True)

# torch.manual_seed(0)
img, label = next(iter(train_dl))
# img = transforms.Resize(5)(img)

#%%
x = img
norm = torch.nn.BatchNorm2d(3, momentum=None)
# norm.weight = torch.nn.Parameter(torch.zeros_like(norm.weight))
y = norm(x)
z = (x[0][0] - torch.mean(x[0][0]))/torch.std(x[0][0])
print(z)
import matplotlib.pyplot as plt

fig = plt.figure(facecolor='w')
for i in range(len(z)):
    plt.scatter(z[i].detach().numpy(),y[0][0][i].detach().numpy())
plt.plot([-3,3],[-3,3])
plt.show()

print(y[0][0])
norm.state_dict()['running_mean']

# print(img.shape)
# print(out.shape)
