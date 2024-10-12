import pytorch_lightning as pl
import json
import torch

from dataset import RandoDataModule
from model import Classifier


results = {}
options = {
    "cus": ["Data/X_Train_cus_imp.csv", "Data/X_Test_cus_imp.csv"],
}
batch_size = 8192

for method, paths in options.items():
    data_module = RandoDataModule(
        batch_size=batch_size,
        x_train_path=paths[0],
        x_test_path=paths[1],
    )
    model = Classifier(lr=0.005, weight_decay=5e-4, batch_size=batch_size)
    # model = Classifier.load_from_checkpoint("cross_attention.ckpt", lr=0.005, weight_decay=1e-4, batch_size=batch_size)

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=30,
        devices=[0],
        log_every_n_steps=1,
    )

    trainer.fit(model, data_module)
    torch.save(model.state_dict(), method + ".pth")
    trainer.test(model, data_module)[0]
    model.conf.plot()

print(json.dumps(results, indent=4))
