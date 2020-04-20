import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets
import os
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import pickle
    
class Discriminator(nn.Module):

    def __init__(self,d):
        super(Discriminator, self).__init__()
        
        self.dconv1 = nn.Conv2d(3, 3, 4, 2, 1,groups = 3)
        self.pconv1 = nn.Conv2d(3, d, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.dconvv1 = nn.Conv2d(d, d, 3, 1, 1, groups=d)
        self.pconvv1 = nn.Conv2d(d, d, 1, 1)
        
        self.dconv2 = nn.Conv2d(d, d, 4, 2, 1,groups = d)
        self.pconv2 = nn.Conv2d(d, d*2, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.dconvv2 = nn.Conv2d(d * 2, d * 2, 3, 1, 1, groups=d * 2)
        self.pconvv2 = nn.Conv2d(d * 2, d * 2, 1, 1)
        
        self.dconv3 = nn.Conv2d(d * 2, d * 2, 4, 2, 1,groups = d * 2)
        self.pconv3 = nn.Conv2d(d * 2, d * 4, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.dconvv3 = nn.Conv2d(d * 4, d * 4, 3, 1, 1, groups=d * 4)
        self.pconvv3 = nn.Conv2d(d * 4, d * 4, 1, 1)
 
        self.dconv4 = nn.Conv2d(d * 4, d * 4, 4, 2, 1,groups = d * 4)
        self.pconv4 = nn.Conv2d(d * 4, d * 8, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.dconvv4 = nn.Conv2d(d * 8, d * 8, 3, 1, 1, groups=d * 8)
        self.pconvv4 = nn.Conv2d(d * 8, d * 8, 1, 1)
        
        self.dconv5 = nn.Conv2d(d * 8, d * 8, 4, 1, 0,groups = d * 8)
        self.pconv5 = nn.Conv2d(d * 8, 1, 1, 1)


    def forward(self,input):
        x = F.leaky_relu(self.conv1_bn(self.pconv1(self.dconv1(input))), 0.2)
        x = F.leaky_relu(self.conv1_bn(self.pconvv1(self.dconvv1(x))), 0.2)
        
        x = F.leaky_relu(self.conv2_bn(self.pconv2(self.dconv2(x))), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.pconvv2(self.dconvv2(x))), 0.2)
        
        x = F.leaky_relu(self.conv3_bn(self.pconv3(self.dconv3(x))), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.pconvv3(self.dconvv3(x))), 0.2)
        
        x = F.leaky_relu(self.conv4_bn(self.pconv4(self.dconv4(x))), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.pconvv4(self.dconvv4(x))), 0.2)
        
        x = F.leaky_relu(self.pconv5(self.dconv5(x)), 0.2)
        x = F.sigmoid(x)
        
        return x

    

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Generator(nn.Module):
    def __init__(self,d):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self,input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


dnet = Discriminator(128)
gnet = Generator(128)
dnet.weight_init(mean=0.0, std=0.02)
gnet.weight_init(mean=0.0, std=0.02)
dnet = dnet.cuda()
gnet = gnet.cuda()
loss = nn.MSELoss().cuda()
doptimizer = optim.Adam(dnet.parameters(),lr=0.01,betas=(0.5,0.999))
goptimizer = optim.Adam(gnet.parameters(),lr=0.001,betas=(0.5,0.999))

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = "这里是放图片的文件夹路径" # the path of the folder that contains the imgs
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=True)

for epoch in range(2000):
    print("epoch %d"%(epoch+1))
    images = []
    d_avgloss = 0
    g_avgloss = 0
    for data,_ in train_loader:
        
        real_y = torch.ones(16,dtype=torch.float).cuda()
        fake_y = torch.zeros(16,dtype=torch.float).cuda()
        
        doptimizer.zero_grad()
        data = Variable(data.cuda())
        
        output = dnet(data).squeeze()
        D_true_loss = loss(output,real_y)
        d_avgloss += D_true_loss.data
        
        ran_input = torch.randn((16,100,1,1)).cuda()
        g_output = gnet(ran_input)
        d_output = dnet(g_output).squeeze()
        D_fake_loss = loss(d_output, fake_y)
        d_avgloss += D_fake_loss.data
        
        D_train_loss = D_true_loss+D_fake_loss
        
        D_train_loss.backward()
        
        doptimizer.step()
        goptimizer.zero_grad()
        ran_input = torch.randn((16, 100, 1, 1)).cuda()
        g_output = gnet(ran_input)
        d_output = dnet(g_output).squeeze()
        G_train_loss = loss(d_output,real_y)
        g_avgloss += G_train_loss.data
        G_train_loss.backward()
        goptimizer.step()
    
        images = g_output
        
    path = "./data/anime_gen/"+str(epoch+1)+".jpg" # 保存图片的路径，请自行修改 you need to change this line on you own

    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(4, 4))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)


    for k in range(4 * 4):
        i = k // 4
        j = k % 4
        ax[i, j].cla()
        ax[i, j].imshow((images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    
    fig.text(0.5, 0.04, "epoch "+str(epoch+1), ha='center')
    plt.savefig(path)
    print("dloss:%.3f gloss:%.3f" %(d_avgloss/32,g_avgloss/16))
    torch.save(gnet.state_dict(),'./pth/gnet.pth')
    torch.save(dnet.state_dict(),'./pth/dnet.pth')

