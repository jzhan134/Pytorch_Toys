#%%
import torch
from torch.utils.data import TensorDataset, DataLoader
data = torch.tensor([[1,1],[2,2],[3,3],[4,4],[5,5]])
label = torch.tensor([11, 12, 13, 14, 15])

# ! zip data and label
dataset = TensorDataset(data, label)

# ! enumerate dataset in batches
# ! There are ceil(sample_nmber / batch_size) iters
# ! batch is in different order every epoch
dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

# ! enumerate list of sample indices in each batch
sampler = dataloader.batch_sampler

for epoch in range(5):
    print(f"Epoch {epoch}:")
    for idx, (x_batch, y_batch) in zip(sampler, dataloader):
        print(f"\tidx: {idx}, data {x_batch.tolist()}, \n")
        # ! If iterate sampler and dataloader in two loops, 
        # ! their indices won't match.