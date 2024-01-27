import torch
from layers.gramian_angular import GramianAngularFieldPytorch
from layers.cs_attention import block_cs_ann
class cs_ann(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.angular_field = GramianAngularFieldPytorch(method="summation")
        self.block_1 = block_cs_ann()
    def forward(self,x):
        x_gramian_angular = self.angular_field(x)
        x = x.unsqueeze(-1)
        y = torch.cat((x_gramian_angular, x), dim=-1).transpose(-3,-1)
        return self.block_1(y)
if __name__=="__main__":
    x = torch.rand(128, 5,64)
    print(cs_ann()(x).size())