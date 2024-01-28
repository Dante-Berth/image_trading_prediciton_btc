import torch
from layers.gramian_angular import GramianAngularFieldPytorch
from layers.cs_attention import block_cs_ann

class cs_ann(torch.nn.Module):
    def __init__(self, config,config_problem=None, channels_first=True,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels_first=channels_first
        self.angular_field = GramianAngularFieldPytorch(method="summation")
        self.config = config
        self.config_problem = config_problem
        self.list_layers = []
        for key in config.keys():
            self.list_layers.append(block_cs_ann(nb_features=config[key]["nb_features"],reduction=config[key]["reduction"], max_pool_reduction=config[key]["max_pool_reduction"]))

        self.sequential_layers = torch.nn.Sequential(*self.list_layers)


    def forward(self,x):
        if self.channels_first:
            x = x.transpose(-2,-1)
        x_gramian_angular = self.angular_field(x)
        x = x.unsqueeze(-1)
        y = torch.cat((x_gramian_angular, x), dim=-1).transpose(-3,-1)
        return self.sequential_layers(y)

if __name__=="__main__":
    x = torch.rand(128, 64, 4)
    config = {
        'layer_1':{
            'nb_features':32,
            'reduction':1,
            "max_pool_reduction":2
        },
        'layer_2':{
            'nb_features':32,
            'reduction':1,
            "max_pool_reduction":2
        },
        'layer_3':{
            'nb_features':32,
            'reduction':2,
            "max_pool_reduction":1
        },
    }
    print(cs_ann(config)(x).size())