import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from model import cs_ann
from torch_sequencer import AtomicSequencer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


class LitModelCs(pl.LightningModule):
    def __init__(self, model, config_pl_lighting):
        super().__init__()
        self.model = model
        self.save_hyperparameters(config_pl_lighting)

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.cross_entropy(z.squeeze(-1), y)/y.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = torch.nn.functional.cross_entropy(z.squeeze(-1), y)/y.size(0)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=5,
                                                         min_lr=5e-8)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss"}

    def train_dataloader(self):
        train_atomic_dataset = AtomicSequencer(PATH=self.hparams["PATH"], begin_date=self.hparams["train"]["begin_date"], end_date=self.hparams["train"]["end_date"])
        train_dataloader = DataLoader(train_atomic_dataset, batch_size=self.hparams["batch_size"],persistent_workers=True, shuffle=True, num_workers=6)
        return train_dataloader
    
    def val_dataloader(self):
        val_atomic_dataset = AtomicSequencer(PATH=self.hparams["PATH"], begin_date=self.hparams["validation"]["begin_date"], end_date=self.hparams["validation"]["end_date"])
        val_dataloader = DataLoader(val_atomic_dataset, batch_size=self.hparams["batch_size"], persistent_workers=True, num_workers=6)
        return val_dataloader

if __name__=="__main__":
    config_pl_lighting = {
        "PATH":r"./data/binance-BTCUSDT-5m.pkl",
        "batch_size":128,
        "learning_rate":1e-4,
        "train":{"begin_date":"2018-05-01",
                 "end_date":"2021-05-01"},
        "validation":{"begin_date":"2021-05-01",
                 "end_date":"2022-11-04"},
    }
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

    
    
    initialiser_atomic_dataset = AtomicSequencer(PATH=config_pl_lighting["PATH"], begin_date="2018-05-01", end_date="2018-05-07")
    initaliser_dataloader = DataLoader(initialiser_atomic_dataset, batch_size=config_pl_lighting["batch_size"], shuffle=True, num_workers=6)
    
    cs_model = cs_ann(config,config_problem=None,channels_first=True)
    mlp_model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.LazyLinear(out_features=32),
    torch.nn.BatchNorm1d(32),
    torch.nn.SiLU(),
    torch.nn.LazyLinear(out_features=64),
    torch.nn.BatchNorm1d(64),
    torch.nn.SiLU(),
    torch.nn.LazyLinear(out_features=16),
    torch.nn.LazyLinear(out_features=1),
    torch.nn.Sigmoid())

    combined_model = torch.nn.Sequential(
        cs_model,
        mlp_model
    )
    
    x,z = next(iter(initaliser_dataloader))
    y = combined_model(x)
    model = LitModelCs(model=combined_model,config_pl_lighting=config_pl_lighting)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    

    trainer = Trainer(enable_progress_bar=True,
                      accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                      max_epochs=150,
                      callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="validation_loss"),
                                 LearningRateMonitor("epoch"),
                                 EarlyStopping(monitor="validation_loss", mode="min", patience=30)])
    tuner = Tuner(trainer)
    model.hparams.lr = tuner.lr_find(model).suggestion()
    
    trainer.fit(model)

