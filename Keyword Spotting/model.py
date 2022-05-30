import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from torch import nn
from transformer import Transformer

class KWSModel(LightningModule):
    def __init__(self, num_classes=37, epochs=1, lr=0.001, depth=12, embed_dim=64,
                 head=4, patch_dim=192, seqlen=16, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Transformer(dim=embed_dim, num_heads=head, num_blocks=depth, mlp_ratio=4.,
                                   qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.embed = torch.nn.Linear(patch_dim, embed_dim)

        self.fc = nn.Linear(seqlen * embed_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        init_weights_vit_timm(self)

    def forward(self, x):
        # Linear projection
        x = self.embed(x)
            
        # Encoder
        x = self.encoder(x)
        x = x.flatten(start_dim=1)

        # Classification head
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        mels, labels, _ = batch
        preds = self(mels)
        loss = self.hparams.criterion(preds, labels)
        return {'loss': loss}

    # calls to self.log() are recorded in wandb BAKA DI KAILANGAN
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        mels, labels, wavs = batch
        preds = self(mels)
        loss = self.hparams.criterion(preds, labels)
        acc = accuracy(preds, labels) * 100.
        return {"preds": preds, 'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.hparams.epochs)
        return [optimizer], [lr_scheduler]

    def setup(self, stage=None):
        self.hparams.criterion = torch.nn.CrossEntropyLoss()

def init_weights_vit_timm(module: nn.Module):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()