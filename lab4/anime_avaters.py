import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os

class anime_dataset(Dataset):
    def __init__(self, root: str, transforms=True) -> None:
        
        def get_anime_names(root: str):
            file_names = []
            for _, __, files in os.walk(root):
                for file in files:
                    file_names.append(root+'/'+file)
            return file_names

        self.file_names = get_anime_names(root)
        self.transforms = transforms

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ])(img)
        return img
    
    def __len__(self):
        return len(self.file_names)

import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    '''
        (N, ) -> (N, 3, 64, 64)
    '''
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, out_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(out_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(out_dim * 8, out_dim * 4, 5, 2,
                    padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim * 4),
            nn.ReLU()
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_dim * 4, out_dim * 2, 5, 2,
                    padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim * 2),
            nn.ReLU()
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(out_dim * 2, out_dim, 5, 2,
                    padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(out_dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.l1(x)
        x1 = x1.view(x1.size(0), -1, 4, 4)
        x2 = self.tconv1(x1)
        x3 = self.tconv2(x2)
        x4 = self.tconv3(x3)
        y  = self.tconv4(x4)
        return y

class Discriminator(nn.Module):
    '''
        (N, 3, 64, 64, ) -> (N, )
    '''
    def __init__(self, in_dim, out_dim):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(out_dim, out_dim * 2),
            conv_bn_lrelu(out_dim * 2, out_dim * 4),
            conv_bn_lrelu(out_dim * 4, out_dim * 8),
            nn.Conv2d(out_dim * 8, 1, 4),
            nn.Sigmoid())
        
    def forward(self, x):
        y = self.ls(x)
        return y.view(-1)

import torch
from torch.autograd import Variable
import torchvision

batch_size = 128
z_dim = 100
lr = 1e-4
n_epoch = 50
root = './faces'

save_dir = os.path.join('./', 'logs')
os.makedirs(save_dir, exist_ok=True)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    G = Generator(z_dim, 64).to(device)
    G.load_state_dict(torch.load('dcgan_g.pth'))
    D = Discriminator(3, 64).to(device)
    D.load_state_dict(torch.load('dcgan_d.pth'))

    G.train()
    D.train()

    criterion = nn.BCELoss()

    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = anime_dataset(root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    z_sample = Variable(torch.randn(100, z_dim)).to(device)

    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.to(device)

            bs = imgs.size(0)

            z = Variable(torch.randn(bs, z_dim)).to(device)
            r_imgs = Variable(imgs).to(device)
            f_imgs = G(z)

            r_label = torch.ones((bs)).to(device)
            f_label = torch.zeros((bs)).to(device)

            r_logit = D(r_imgs)
            f_logit = D(f_imgs)

            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = Variable(torch.randn(bs, z_dim)).to(device)
            f_imgs = G(z)

            f_logit = D(f_imgs)

            loss_G = criterion(f_logit, r_label)

            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
        
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(save_dir, f'Epoch_{epoch+60:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        G.train()
        if (e+1) % 5 == 0:
            torch.save(G.state_dict(), os.path.join('./', f'dcgan_g.pth'))
            torch.save(D.state_dict(), os.path.join('./', f'dcgan_d.pth'))

if __name__ == '__main__':
    main()