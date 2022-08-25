#%%
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ! Setters
x = torch.tensor([1,2,3,4], device=device)
y = torch.tensor([4,5,6]).to(device)
z = torch.tensor([4,5,6]).cuda()
model = torch.nn.Linear(2,2,device=device)
model.to(device)
x.is_cuda # True|False