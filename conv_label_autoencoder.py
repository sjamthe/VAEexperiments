#original author = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import time
import inspect
from torchviz import make_dot

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# mps is 4x faster than CPU on M1

torch.manual_seed(10)

dir = './conv_img/'
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

#img_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 10, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(90,10)
            #nn.MaxPool2d(3, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 16, 5, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        code = self.encoder(x)
        output = self.decoder(code.unsqueeze(-1).unsqueeze(-1))
        return code, output


model = conv_autoencoder()
num_params = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_params}')
if torch.cuda.is_available():
    model.cuda()
else:
    model.to(device)

criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(1,num_epochs+1):
    t0 = time.time()
    for data in dataloader:
        img, label = data
        #img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        else:
            img = img.to(device)
        # ===================forward=====================
        #code = model.encoder(img)
        code, output = model(img)
        label_loss = criterion2(code, label.to(device))
        recon_loss = criterion(output, img)
        loss = recon_loss + label_loss
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         # Generate a Graphviz object from the computation graph
        #graph = make_dot(loss, params=dict(model.named_parameters())) 

        # Save the graph as a PDF or any other format if needed
        #graph.render("conv_autoencoder_graph")
    # ===================log========================
    t1 = time.time()
    print('epoch [{}/{}], lr:{:.5f} loss:{:.4f}, label_loss:{:.4f}, \
          recon_loss:{:.4f}, step time {:.2f}ms'
          .format(epoch, num_epochs,learning_rate, loss.item(),
            label_loss.item(), recon_loss.item(), (t1-t0)*1000))

    
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, dir + 'image_{}.png'.format(epoch))
        save = to_img(img.cpu().data)
        save_image(save, dir + 'orig_image_{}.png'.format(epoch))

        torch.save(model.state_dict(), './conv_label_autoencoder.pth')

        class_code = inspect.getsource(conv_autoencoder)
        with open('conv_label_autoencoder_class.py', 'w') as f:
            f.write(class_code)
