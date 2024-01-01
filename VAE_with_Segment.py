# Implement VAE with Segment

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import time
from segment import Segment
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    #else "mps"
    #if torch.backends.mps.is_available()
    else "cpu"
)
# CPU is 2x faster on M1 than mps for this model
print(f"Using {device} device")

torch.manual_seed(10)

dir = './vae_seg_img/'
if not os.path.exists(dir):
    os.mkdir(dir)

def yact_to_img(x):
    mask = x > x.mean()
    x[mask] = 1
    x[x <1] = 0
    return to_img(x)

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def img_to_yact(x):
    """ convert gray scale image to image which has values of y-position instead of grayscale
    """
    test_image = x.numpy() 
    limit = test_image.mean()
    bw_array  = torch.tensor(np.where(test_image < limit, 0, 1))
    [_, _, height, width] = bw_array.shape
    xmax = width + 1
    ymax = height + 1
    maxval = max(xmax, ymax)
    y_in = torch.arange(1./maxval, ymax/maxval, 1/maxval)
    #create array that contains y pos values instead of 1
    yval_array = y_in.view(y_in.shape[0], 1)*bw_array
    return yval_array

num_epochs = 100
batch_size = 128
learning_rate = 1e-4

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE_Seg(nn.Module):
    def __init__(self):
        super(VAE_Seg, self).__init__()

        #self.seg1 = Segment(784, 20, 2)
        #self.seg21 = Segment(400, 20, 4)
        #self.seg22 = Segment(400, 20, 4)
        #self.seg3 = Segment(20, 400, 4)
        #self.fc21 = nn.Linear(20, 10)
        #self.fc22 = nn.Linear(20, 10)
        #self.fc3 = nn.Linear(10, 20)
        #self.seg4 = Segment(20, 784, 2)
        #Originsl
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
            eps = eps.to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE_Seg()
if torch.cuda.is_available():
    model.cuda()
else:
    model.to(device)

reconstruction_function = nn.MSELoss(reduction='sum')


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0
    t0 = time.time()
    for batch_idx, data in enumerate(dataloader):
        orig_img, _ = data
        img = img_to_yact(orig_img)
        img = img.view(img.size(0), -1)
        orig_img = orig_img.view(orig_img.size(0), -1)
        
        if torch.cuda.is_available():
            img = img.cuda()
        else:
            img = img.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img),
                len(dataloader.dataset), 100. * batch_idx / len(dataloader),
                loss.item() / len(img)))

    t1 = time.time()
    print('====> Epoch: {} Average loss: {:.4f} | step time {:.2f}ms'.format(
        epoch, train_loss / len(dataloader.dataset), ((t1-t0)*1000)))
    if epoch % 10 == 0:
        save = to_img(recon_batch.cpu().data)
        save_image(save, dir + 'image_{}.png'.format(epoch))
        save = to_img(orig_img.cpu().data)
        save_image(save, dir + 'yact_image_{}.png'.format(epoch))

print('Final Epoch: {} | Average loss: {:.4f} | Step time {:.2f}ms | Learning Rate: {:.5f}'.format(
        epoch, train_loss / len(dataloader.dataset), ((t1-t0)*1000), learning_rate))
torch.save(model.state_dict(), './vae_seg.pth')

###### TRAINING STATS ########
### lr = 1e-4
### ====> Epoch: 99 Average loss: 28.1472 | step time 3630.71ms  | Learning Rate: .0004