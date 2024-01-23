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

class SpatialAttentionLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_shape = None
        self.lazy_linear_query = None
        self.lazy_linear_key = None
        self.lazy_linear_value = None
        self.lazy_conv2d_1 = None
        self.lazy_conv2d_2 = None

    def build(self, x)->None:
        self.input_shape = x.size()
        self.lazy_conv2d_1 = torch.nn.LazyConv2d(self.input_shape[-3],1)
        self.lazy_conv2d_2 = torch.nn.LazyConv2d(self.input_shape[-2],1)
        self.lazy_linear_query = torch.nn.LazyLinear(self.input_shape[-1])
        self.lazy_linear_key = torch.nn.LazyLinear(self.input_shape[-1])
        self.lazy_linear_value = torch.nn.LazyLinear(self.input_shape[-1])

    @staticmethod
    def reshape_3d(x:torch.Tensor)->torch.Tensor:
        """
        Convert a tensor (B,C,H,W) into (B,C*H,W)
        """
        x_shape = x.size()
        return torch.reshape(x,(x_shape[0],x_shape[-3]*x_shape[-2],x_shape[-1]))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.input_shape is None:
            self.build(x)
        q = self.reshape_3d(self.lazy_linear_query(x))
        k = self.reshape_3d(self.lazy_linear_key(x))
        v = self.reshape_3d(self.lazy_linear_value(x))
        qk = torch.matmul(q, k.transpose(-2,-1))
        attention_weights = F.softmax(qk, dim=-1)
        qkv = torch.matmul(attention_weights, v)
        z = self.lazy_conv2d_1(torch.unsqueeze(qkv,dim=1))
        z = z.transpose(-3,-2)
        z = self.lazy_conv2d_2(z)
        return z

class SpatialAttentionConvolution(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_shape = None
        self.lazy_conv2d_query = torch.nn.LazyConv2d(1,1)
        self.lazy_conv2d_key = torch.nn.LazyConv2d(1,1)
        self.lazy_conv2d_value = torch.nn.LazyConv2d(1,1)
        self.lazy_conv2d_2 = None

    def build(self, x)->None:
        self.input_shape = x.size()
        self.lazy_conv2d_2 = torch.nn.LazyConv2d(self.input_shape[-3],1)

    @staticmethod
    def reshape_3d(x:torch.Tensor)->torch.Tensor:
        """
        Convert a tensor (B,C,H,W) into (B,C*H,W)
        """
        x_shape = x.size()
        return torch.reshape(x,(x_shape[0],x_shape[-3]*x_shape[-2],x_shape[-1]))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.input_shape is None:
            self.build(x)
        q = self.reshape_3d(self.lazy_conv2d_query(x))
        k = self.reshape_3d(self.lazy_conv2d_key(x))
        v = self.reshape_3d(self.lazy_conv2d_value(x))
        qk = torch.matmul(q, k.transpose(-2,-1))
        attention_weights = F.softmax(qk, dim=-1)
        qkv = torch.matmul(attention_weights, v)
        z = self.lazy_conv2d_2(torch.unsqueeze(qkv,dim=1))
        return z


if __name__=="__main__":
    input_image_size = torch.rand(128, 20, 20, 5)
    Fc = ChannelAttention(64)(input_image_size)
    print((SpatialAttentionLinear()(Fc) + Fc + SpatialAttentionConvolution()(Fc)).size() == Fc.size())
    
