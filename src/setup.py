import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from model import cs_ann
from torch_sequencer import AtomicSequencer

class LitModelCs(pl.LightningModule):
    def __init__(self, model, PATH=r"./data/binance-BTCUSDT-5m.pkl",batch_size=128,learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.model = model
        self.batch_size = batch_size
        self.PATH = PATH
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.cross_entropy(z.squeeze(-1), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.cross_entropy(z.squeeze(-1), y)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_atomic_dataset = AtomicSequencer(PATH=PATH, begin_date="2018-05-01", end_date="2021-11-03")
        train_dataloader = DataLoader(train_atomic_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)
        return train_dataloader

if __name__=="__main__":
    PATH = r"./data/binance-BTCUSDT-5m.pkl"
    config_problem = None
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

    batch_size = 128
    
    train_atomic_dataset = AtomicSequencer(PATH=PATH, begin_date="2018-05-01", end_date="2021-11-03")
    train_dataloader = DataLoader(train_atomic_dataset, batch_size=batch_size, shuffle=True, num_workers=6)


    cs_model = cs_ann(config,config_problem=None,channels_first=True)
   

    mlp_model = torch.nn.Sequential(
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
    
    x,z = next(iter(train_dataloader))
    y = combined_model(x)
    from pytorch_lightning.tuner.tuning import Tuner
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    model = LitModelCs(model=combined_model)
       
    
    val_atomic_dataset = AtomicSequencer(PATH=PATH, begin_date="2021-11-03", end_date="2022-11-04")
    val_dataloader = DataLoader(val_atomic_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    
    trainer = Trainer(enable_progress_bar=True,callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=30)])
    tuner = Tuner(trainer)
    model.hparams.lr = tuner.lr_find(model).suggestion()
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

