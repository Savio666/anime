import torch
from torch import nn



class Generator(nn.Module):
    def __init__(self,z_dim=10,channels=3,hidden_dim=64):
        super(Generator,self).__init__()
        self.z_dim=z_dim
        self.gen = nn.Sequential(
            self.gen_block(z_dim,hidden_dim*4),
            self.gen_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            self.gen_block(hidden_dim*2, hidden_dim),
            self.gen_block(hidden_dim,channels,kernel_size=4,final_layer=True)
        )

    def gen_block(self,input_channels,output_channels, kernel_size=3, stride=2,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                    nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride),
                nn.Tanh()
            )

    def forward(self,noise):
         x = noise.view(len(noise), self.z_dim, 1, 1)
         return self.gen(x)

def get_noise(n_samples,z_dim,device="cpu"):
    return torch.randn(n_samples,z_dim,device=device)

class Critic(nn.Module):
    def __init__(self,channels=3,hidden_dim=64):
        super(Critic, self).__init__()
        self.crit= nn.Sequential(
                self.crit_block(channels,hidden_dim),
                self.crit_block(hidden_dim,hidden_dim*2),
                self.crit_block(hidden_dim*2,1,final_layer=True)
            )

    def crit_block(self,input_channels,output_channels,kernel_size=4,stride=2,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                    nn.Conv2d(input_channels,output_channels,kernel_size,stride),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(0.2, inplace = True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
            )

    def forward(self,image):
        output=self.crit(image)
        return output.view(len(output),-1)