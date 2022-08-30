#%%
if __name__ == '__main__':
    import os
    from turtle import forward
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as T
    from torchvision.utils import save_image
    import torch.nn.functional as F
    from tqdm import tqdm
    torch.manual_seed(10)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    latent_size = 128
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_ds = ImageFolder(
        r'animefacedataset', 
        transform=T.Compose([
            T.Resize(64), T.CenterCrop(64), T.ToTensor(),
            T.Normalize(*stats)])
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    #%% # ! Discriminator and generator models
    import torch.nn as nn
    discriminator = nn.Sequential(# in: 3 x 64 x 64
        nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True), # out: 64 x 32 x 32
        # ! Leaky ReLU makes the gradients from the discriminator 
        # ! flows stronger into the generator
        
        nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True), # out: 128 x 16 x 16

        nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True), # out: 256 x 8 x 8

        nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True), # out: 512 x 4 x 4

        nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),

        nn.Flatten(),
        nn.Sigmoid() # out: 1 x 1 x 1
    )
    discriminator.to(dev)


    generator = nn.Sequential( # in: latent_size x 1 x 1
        nn.ConvTranspose2d(latent_size, 512, 4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True), # out: 512 x 4 x 4

        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True), # out: 256 x 8 x 8

        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True), # out: 128 x 16 x 16

        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True), # out: 64 x 32 x 32

        nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
        nn.Tanh() # out: 3 x 64 x 64
    )
    generator.to(dev)




    #%% # ! helper functions
    # ! This has two parts. First, use real images with target = 1
    # ! as positive learnings. Secon,d use generator to generate
    # ! fake images with target = 0 as negative learnings.
    # ! Use both to update discriminator.
    def train_discriminator(r_imgs, opt_d):
        opt_d.zero_grad()
        r_pred = discriminator(r_imgs)
        r_target = torch.ones(r_imgs.size(0),1,device=dev)
        r_loss = F.binary_cross_entropy(r_pred, r_target)
        r_score = torch.mean(r_pred).item()

        latent = torch.rand(batch_size, latent_size, 1, 1, device=dev)
        f_imgs = generator(latent)
        f_pred = discriminator(f_imgs)
        f_target = torch.zeros(f_imgs.size(0), 1, device=dev)
        f_loss = F.binary_cross_entropy(f_pred, f_target)
        f_score = torch.mean(f_pred).item()

        loss = r_loss + f_loss
        loss.backward()
        opt_d.step()
        return loss.item(), r_score, f_score

    # ! Generate fake images and let discriminator make prediction.
    # ! Then cheat it that the images are real. 
    # ! Then the loss is the gap between generated fake images
    # ! and true real images.
    # ! The more accurate the discriminator is, the more efficient 
    # ! generator could learn.
    def train_generator(opt_g):
        opt_g.zero_grad()
        latent = torch.randn(batch_size, latent_size, 1, 1, device=dev)
        f_imgs = generator(latent)
        preds = discriminator(f_imgs)
        targets = torch.ones(batch_size, 1, device=dev)
        loss = F.binary_cross_entropy(preds, targets)
        loss.backward()
        opt_g.step()
        return loss.item()

    # * save fake generated images
    def save_samples(index, latent_tensors):
        f_imgs = generator(latent_tensors)
        f_pred = discriminator(f_imgs)
        print(torch.mean(f_pred).item())
        f_name = 'generated-images-{0:0=4d}.png'.format(index)
        f_imgs = f_imgs * stats[1][0] + stats[0][0]
        save_image(f_imgs, os.path.join("Results", f_name), nrow=8)

    # * The same latent to generate saved samples
    fixed_latent = torch.randn(1, latent_size, 1, 1, device=dev)

    #%% # ! Training loop
    # torch.cuda.empty_cache()
    opt_d = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr = 0.0002, betas=(0.5, 0.999))

    for epoch in range(25):
        cnt = 0
        for r_imgs, _ in tqdm(train_dl, total=50):
            cnt += 1
            r_imgs = r_imgs.to(dev,non_blocking=True)
            loss_d, r_score_d, f_score_d = train_discriminator(r_imgs, opt_d,)
            loss_g = train_generator(opt_g)
            if cnt > 50:
                break

        print(
            f"Epoch: {epoch} " +
            f"loss_g: {loss_g:.4f} " +
            f"loss_d: {loss_d:.4f} " +
            f"r_score_d: {r_score_d:.4f} " +
            f"f_score_d: {f_score_d:.4f} "
        )
        save_samples(epoch, fixed_latent)