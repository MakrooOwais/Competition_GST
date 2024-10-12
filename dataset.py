import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class RandoDataset(Dataset):
    def __init__(self, path_x, path_y, means, stds, train=False):
        super().__init__()
        self.path_x = path_x
        self.path_y = path_y
        self.train = train
        self.means = torch.Tensor(means)
        self.stds = torch.Tensor(stds)
        self.X = pd.read_csv(path_x)
        print("X Loaded!!")
        self.Y = pd.read_csv(path_y)
        print("Y Loaded!!")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.Tensor(self.X.iloc[index].to_list()[1:])
        Y_ = self.Y.iloc[index].to_list()[1]
        Y = torch.zeros(2)
        Y[Y_] = 1
        X = (X - self.means) / self.stds
        data = {"X": X, "Y": Y}
        return data


class RandoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, x_train_path, x_test_path):
        super().__init__()
        self.batch_size = batch_size
        self.x_train_path = x_train_path
        self.x_test_path = x_test_path

    def prepare_data(self):
        pass

    def setup(self, stage):
        means, stds = list(), list()
        x = pd.read_csv(self.x_train_path)
        for i in range(22):
            means.append(x["Column" + str(i)].mean())
            stds.append(x["Column" + str(i)].std())

        self.train = RandoDataset(
            self.x_train_path, "Data/Y_Train_Data_Target.csv", means, stds, train=True
        )
        self.val = RandoDataset(
            self.x_test_path, "Data/Y_Test_Data_Target.csv", means, stds
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":
    dataset = RandoDataset("Data/X_Train_NONAN.csv", "Data/Y_Train_Data_Target.csv")
    print(len(dataset))
    print(dataset[0])
