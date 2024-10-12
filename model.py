import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC


class Classifier(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, weight_decay: float = 5e-3, batch_size: int = 512
    ):
        super(Classifier, self).__init__()
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        bias = False
        self.model = nn.Sequential(
            nn.Linear(22, 512, bias=bias),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512, bias=bias),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256, bias=bias),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=bias),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.acc = Accuracy("binary")
        self.prec = Precision("binary")
        self.rec = Recall("binary")
        self.f1 = F1Score("binary")
        self.auc_roc = AUROC("binary")
        self.lr = lr
        self.test_outputs = list()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch["X"], batch["Y"]
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, targets)
        acc = self.acc(outputs, targets)

        self.log_dict(
            {"train_loss": loss, "train_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "targets": targets, "outputs": outputs}

    def validation_step(self, batch, batch_idx):
        data, targets = batch["X"], batch["Y"]
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, targets)
        acc = self.acc(outputs, targets)
        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "targets": targets, "outputs": outputs}

    def test_step(self, batch, batch_idx):
        data, targets = batch["X"], batch["Y"]
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, targets)
        acc = self.acc(outputs, targets)
        prec = self.prec(outputs, targets)
        rec = self.rec(outputs, targets)
        f1 = self.f1(outputs, targets)
        auc_roc = self.auc_roc(outputs, targets)

        self.log_dict(
            {
                "test_loss": loss,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1,
                "auc_roc": auc_roc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.test_outputs.append({"targets": targets, "outputs": outputs})
        return {"loss": loss, "targets": targets, "outputs": outputs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,  # , momentum=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, (785133 // self.batch_size) + 1
                ),
            },
        }
