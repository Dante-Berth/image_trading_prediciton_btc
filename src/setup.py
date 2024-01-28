import pytorch_lightning
import torch
import pytorch_lightning as pl
from model import cs_ann
from torch_sequencer import AtomicSequencer
from torch.utils.data import DataLoader
class LitModelCs(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.validation_step_outputs.append(loss)
        self.log("validation_loss", self.validation_step_outputs[-1], on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__=="__main__":
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
    cs_model = cs_ann(config)

    mlp_model = torch.nn.Sequential(
    torch.nn.MaxPool2d(2,2),
    torch.nn.Flatten(),
    torch.nn.LazyLinear(out_features=32),
    torch.nn.BatchNorm1d(32),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(out_features=64),
    torch.nn.BatchNorm1d(64),
    torch.nn.Tanh(),
    torch.nn.LazyLinear(out_features=16),
    torch.nn.LazyLinear(out_features=1),
    torch.nn.Sigmoid())

    combined_model = torch.nn.Sequential(
        cs_model,
        mlp_model
    )
    
    PATH = r"./data/binance-BTCUSDT-5m.pkl"
     
    train_atomic_dataset = AtomicSequencer(PATH=PATH, begin_date="2018-05-01", end_date="2021-11-03")
    val_atomic_dataset = AtomicSequencer(PATH=PATH, begin_date="2021-11-03", end_date="2022-11-03")
    batch_size = 128
    train_dataloader = DataLoader(train_atomic_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_atomic_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print("Model")
    x = next(iter(train_dataloader))[0]
    y = combined_model(x)
    model = LitModelCs(model=combined_model)
    print("Begging")
    trainer = pytorch_lightning.Trainer(enable_progress_bar=True)#,callbacks=[MyCallback()])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

