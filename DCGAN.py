import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def show_result(G,num_epoch,fixed_z_, show = False, path = 'result.png',isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
def train_GAN():
    # training parameters
    lr = 0.0002
    train_epoch = 50
    
    fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1).cuda()    # fixed noise
    # data_loader
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    data_dir = "保存图片的文件夹的路径" # the path of the folder that contains the imgs
    
    dset = datasets.ImageFolder(data_dir, transform)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=True)
    
    # network
    G = generator(128)
    D = discriminator(128)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()
    
    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()
    
    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    
    print('Training start!')
    for epoch in range(train_epoch):
        D_losses = 0
        G_losses = 0
        
        # learning rate decay
        # if (epoch + 1) == 11:
        #     G_optimizer.param_groups[0]['lr'] /= 10
        #     D_optimizer.param_groups[0]['lr'] /= 10
        #     print("learning rate change!")
        #
        # if (epoch + 1) == 16:
        #     G_optimizer.param_groups[0]['lr'] /= 10
        #     D_optimizer.param_groups[0]['lr'] /= 10
        #     print("learning rate change!")
        
        num_iter = 0
        
        epoch_start_time = time.time()
        for x_, _ in train_loader:
            # train discriminator D
            D.zero_grad()
            
            mini_batch = x_.size()[0]
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)
            
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
            G_result = G(z_)
            
            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            
            D_train_loss = D_real_loss + D_fake_loss
            
            D_train_loss.backward()
            D_optimizer.step()
            
            D_losses += D_train_loss.data
            
            # train generator G
            G.zero_grad()
            
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
            
            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()
            
            G_losses+=G_train_loss.data
            
            num_iter += 1
            
        torch.save(G.state_dict(), './pth/DCGAN_g.pth')
        torch.save(D.state_dict(), './pth/DCGAN_d.pth')
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        p = 'data/anime_dcgen/' + str(epoch + 1) + '.png' # you need change these two lines on your own
        fixed_p = 'data/anime_dcgen/' + "_"+str(epoch + 1) + '.png'
        show_result(G,(epoch + 1),fixed_z_,  path=p, isFix=False)
        show_result(G,(epoch + 1),fixed_z_, path=fixed_p, isFix=True)
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime,D_losses/80,
                                                                     G_losses/80))

        
train_GAN()
