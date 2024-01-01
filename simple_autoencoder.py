__author__ = 'SherlockLiao'

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import time

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

dir = './mlp_img/'
if not os.path.exists(dir):
    os.mkdir(dir)


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64), 
            nn.ReLU(True), 
            nn.Linear(64, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28), nn.Tanh())

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x1, x2


model = autoencoder()
num_params = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_params}')
if torch.cuda.is_available():
    model.cuda()
else:
    model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(1,num_epochs+1):
    t0 = time.time()
    for data in dataloader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        else:
            img = img.to(device)
        # ===================forward=====================
        code, output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    t1 = time.time()
    print('epoch [{}/{}], lr:{:.5f} loss:{:.4f}, step time {:.2f}ms'
          .format(epoch, num_epochs,learning_rate, loss.item(),(t1-t0)*1000))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, dir + 'image_{}.png'.format(epoch))
        save = to_img(img.cpu().data)
        save_image(save, dir + 'orig_image_{}.png'.format(epoch))
        # show code
        #print(f"label {label}, code {code}")

        torch.save(model.state_dict(), './sim_autoencoder.pth')
        with open('sim_autoencoder_class.py', 'w') as f:
            f.write(model.__class__.__code__.co_code)
