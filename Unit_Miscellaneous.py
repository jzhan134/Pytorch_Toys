#%% # ! adaptive average pool always end up 
import torch
x = torch.vstack(
    (
        torch.arange(0,5,dtype=torch.float32), 
        torch.arange(10,15,dtype=torch.float32)
    )
).unsqueeze(0) # * 1 * 2 * 5
model = torch.nn.AdaptiveAvgPool1d(output_size=(3))
print(x)
print(model(x))


#%% # ! customize model paramters
import torch
model = torch.nn.AdaptiveAvgPool1d(output_size=(3))
model[0].weight = torch.nn.Parameter(torch.ones_like(model[0].weight))
model[0].bias = torch.nn.Parameter(torch.zeros_like(model[0].bias))


#%% # ! Save and load model
model = torch.nn.Linear(10,1)
torch.save(model, r'xx/xxx.pt')
model_new = torch.load(r'xx/xxx.pt')


#%% # ! Copy a model or optimizer
modelA = torch.nn.Linear(10,1)
sd = modelA.state_dict()
modelB = torch.nn.Linear(10,1)
modelB.load_state_dict(sd)


#%% # ! Transpose & Padding
x = torch.tensor([[1.,1.,1.,1.,1.,1.,1.,1.]])
conv = torch.nn.Conv1d(1,1,4,stride=2,padding=1) # * Padding add each side
y = conv(x) # y.shape = 1*4
trans = torch.nn.ConvTranspose1d(1,1,4, stride=2, padding=1)
# ! conv and ConvTranspose exactly match (mirror) each other.
# ! stride and padding parameters are exactly the same.
trans_x = trans(y) # * trans_x.shape=8