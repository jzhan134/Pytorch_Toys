#%% 1D Linear Regression 
###################################################################


import numpy as np
import torch
import matplotlib.pyplot as plt


torch.manual_seed(10) # Fix seed for reproducibility
data_size = 100
x = torch.arange(data_size, dtype=torch.float32).reshape(-1,1)
x += torch.rand((data_size,1), dtype=torch.float32)
y = torch.arange(data_size, dtype=torch.float32).reshape(-1,1)
y += 5 * torch.rand((data_size,1), dtype=torch.float32)

#%% Using nn.Linear
###################################################################
m = torch.nn.Linear(in_features=1, out_features=1) 
# bias=False for 0 intercept
# nn.module can be override to customize model
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(m.parameters(), lr = 0.00001)

errs = []
for epoch in range(100):
    pred = m(x) # linear fit
    loss = criterion(pred, y) # difference
    # reset existing gradient, or new gradient will be added
    optimizer.zero_grad() 
    loss.backward() # calculate gradient.
    optimizer.step() # gradient descent update
    errs.append(loss.item())



#%% Using manual operations
###################################################################
w = torch.randn(1,dtype=torch.float32, requires_grad=True)
b = torch.randn(1, dtype=torch.float32, requires_grad=True)
errs = []
for epoch in range(100):
    pred = w*x + b
    loss = torch.mean((pred-y)**2)
    loss.backward()
    errs.append(loss.item())
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

pred = w*x + b


# %% Display results
###################################################################
fig,[ax0, ax1] = plt.subplots(1,2,facecolor='w',figsize=(6,3))
ax0.scatter(x,y,c='r',label='Raw',s=1)
pred = m(x)
ax0.plot(
    x,pred.detach().numpy(),c='b',
    label=f'Fit (loss = {errs[-1]:.2f}'
)
ax0.legend()
ax0.set_xlabel('x')
ax0.set_ylabel('y')

ax1.scatter(np.arange(100),errs,c='k',s=1)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
plt.tight_layout()
plt.show()
# %%