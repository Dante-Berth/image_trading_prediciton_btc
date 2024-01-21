import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self,nb_features:int=64,reduction:int=2):
        super(ChannelAttention, self).__init__()

        # Variables
        self.nb_features = nb_features
        self.reduction = reduction
        # Layers
        self.adamaxpool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.adameanpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.lazy_linear_1 = torch.nn.LazyLinear(self.nb_features)
        self.lazy_linear_2 = torch.nn.LazyLinear(self.nb_features)
        self.max_pool = torch.nn.MaxPool2d(self.reduction,self.reduction)
        self.conv_2d = torch.nn.LazyConv2d(self.nb_features,1)
        self.soft_max = torch.nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor)->torch.Tensor:

        y = torch.transpose(x, dim0=-1, dim1=-3) 
        y = self.conv_2d(y)   
        y = self.max_pool(y)

        z_max = self.adamaxpool(y)
        z_max = torch.transpose(z_max, dim0=-1, dim1=-3)
        z_max = self.lazy_linear_1(z_max)

        z_mean = self.adameanpool(y)
        z_mean = torch.transpose(z_mean, dim0=-1, dim1=-3)
        z_mean = self.lazy_linear_2(z_mean)

        z_final = self.soft_max(z_mean + z_max)

        y = torch.transpose(y, dim0=-1, dim1=-3) 

        return y*z_final




if __name__=="__main__":
    input_image_size = torch.rand(128, 20, 20, 5)
    Fc = ChannelAttention(32)(input_image_size)
    print(Fc.size())
    z = torch.nn.LazyConv2d(1,1)(Fc)
    print(z.size())
    z_f = torch.reshape(z,(z.size(0),z.size(1)*z.size(2),z.size(3)))
    z_f2 = torch.matmul(z_f,z_f.transpose(-2, -1))
    attention_weights = F.softmax(z_f2, dim=-1)
    output = torch.matmul(attention_weights, z_f)
    output = torch.nn.LazyConv2d(10,1)(torch.unsqueeze(output,dim=1))
    print(output.size())
    print((output+Fc).size())
    
