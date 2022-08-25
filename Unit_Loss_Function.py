#%% 
# * binary cross entropy & cross entropy
'''
CE 
    multi-class classification
    CrossEntropyLoss = softmax + log + cross entropy
    input.shape = (N,C), 
    target.shape = (N) for class indices or (N,C) for probabilities
BCE 
    binary classification (special case of CE when C=2)
    BCEWithLogitsLoss = Sigmoid + BCELoss 
        sigmoid is a special case of softmax
    input.shape = (N)
    target.shape = (N)


softmax (and sigmoid) transform raw prediction value to a 
normalized value. It is not a probability.
Softmax is better than normalization for its numerical Stability, 
cheaper Model training cost, and penalises Larger error.
'''
import torch

torch.manual_seed(10)
input = torch.randn(3)
target = torch.empty(3).random_(2)
loss_BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
print(loss_BCE(input, target).data) # input needs to reframe to 0 and 1

input = torch.randn(3, 5) # 3 samples with 5 classes
target = torch.randn(3, 5).softmax(dim=1)
loss_CE = torch.nn.CrossEntropyLoss(reduction='none')
print(loss_CE(input, target).data)



# %%
# * Mean Absolute Error (MAE) & Mean Squared Error (MSE)
import torch
torch.manual_seed(10)
input = torch.randn(3)
target = torch.empty(3).random_(2)
print(f"{input.data=}")
print(f"{target.data=}")

loss_MAE = torch.nn.L1Loss(reduction='none')
print(f"loss_MAE: {loss_MAE(input, target).data.tolist()}")

loss_MSE = torch.nn.MSELoss(reduction='none') # not the square root
print(f"loss_MSE: {loss_MSE(input, target).data.tolist()}")