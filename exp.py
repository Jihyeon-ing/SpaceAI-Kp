import numpy as np

from dataloader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import tqdm
from models import *
import os
import matplotlib.pyplot as plt
from torchsummary import summary

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("CUDA available:", torch.cuda.is_available())
        print("Using device:", self.device)
        if self.device.type == 'cuda':
            print("GPU name:", torch.cuda.get_device_name(0))

        self.model_path = f'./models/{self.args.mode}/{self.args.ex_name}'
        os.makedirs(self.model_path, exist_ok=True)

        self.cp_path = os.path.join(self.model_path, self.args.model_name, self.args.cp_path)
        os.makedirs(self.cp_path, exist_ok=True)

    def model(self):
        if self.args.model_name == 'tsmixer':
            model = TSMixer(
                input_length=self.args.input_len,
                input_channels=self.args.n_features,
                output_length=self.args.tar_len,
                hidden_dim=self.args.hidden_dim,
                n_blocks=self.args.n_blocks,
                dropout_rate=self.args.dropout
            ).to(self.device)

        elif self.args.model_name == 'mlp':
            model = simpleMLP(input_length=self.args.input_len,
                              input_channels=self.args.n_features,
                              output_length=self.args.tar_len,
                              hidden_dim=self.args.hidden_dim
                              ).to(self.device)

        else:
            raise ValueError

        summary(model, (self.args.input_len, self.args.n_features))
        return model

    def weighted_mse_loss(self, pred, tar):
        mse_weight = tar
        mse =  F.mse_loss(pred, tar, reduction='none')
        mse_mean = (mse * mse_weight).mean()
        mse_max = mse.max()
        loss = self.args.alpha * mse_mean + (1 - self.args.alpha) * mse_max
        return loss

    def train(self):
        model = self.model().to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=5)
        train_dataset = CustomDataset(mode=self.args.mode, flag='train')
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)

        val_dataset = CustomDataset(mode=self.args.mode, flag='test')
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.args.batch_size,
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)

        train_losses = []
        val_losses = []

        for epoch in range(1, self.args.epochs + 1):
            model.train()
            running_loss = 0.0

            loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)
            for x, y in loop:
                x, y = x.to(self.device)[:, -self.args.input_len:, :], y.to(self.device)

                optimizer.zero_grad()
                pred = model(x)
                loss = self.weighted_mse_loss(pred, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            print(f"[Epoch {epoch:03d}/{self.args.epochs:03d}] loss: {epoch_loss:.6f}")

            # ------ validation ------ #
            model.eval()

            val_loss = 0.0
            with torch.no_grad():
                vloop = tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch", leave=False)
                for x_val, y_val in vloop:
                    x_val, y_val = x_val.to(self.device)[:, -self.args.input_len:, :], y_val.to(self.device)
                    pred_val = model(x_val)
                    loss_val = criterion(pred_val, y_val)
                    val_loss += loss_val.item() * x_val.size(0)
                    vloop.set_postfix(loss=loss_val.item())

            val_loss /= len(val_loader)

            if epoch > 1:
                if val_loss < min(val_losses):
                    print(f"At epoch {epoch}, val loss decreases from {min(val_losses):.4f} to {val_loss:.4f}.")

            val_losses.append(val_loss)
            print(f"[Epoch {epoch:03d}/{self.args.epochs:03d}] val loss: {val_loss:.6f}")

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
            ckpt_path = os.path.join(self.cp_path, f"ckpt_epoch_{epoch:02d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'input_lenght': self.args.input_len,
                'weight': self.args.alpha,
                'hidden_dim': self.args.hidden_dim,
                'n_blocks': self.args.n_blocks,
                'dropout_rate': self.args.dropout
            }, ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}\n")


        epochs = range(1, self.args.epochs + 1)
        plt.figure()
        plt.plot(epochs, train_losses,       label='Train Loss')
        plt.plot(epochs, val_losses,         label='Validation Loss')
        plt.title(f'Training vs Validation Loss, best epoch: {np.argmin(val_losses)+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test(self):
        pth_file = os.path.join(self.cp_path, f"ckpt_epoch_{self.args.test_epoch:02d}.pth")
        ckpt = torch.load(pth_file, map_location=self.device)
        model = TSMixer(
            input_length=self.args.input_len,
            input_channels=self.args.n_features,
            output_length=self.args.tar_len,
            hidden_dim=ckpt['hidden_dim'],
            n_blocks=ckpt['n_blocks'],
            dropout_rate = ckpt['dropout_rate'],
        ).to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"======= hyperpameters of {self.args.ex_name} =======")
        print("hidden_dim: ", ckpt['hidden_dim'])
        print("weight: ", ckpt['weight'])
        print("n_blocks: ", ckpt['n_blocks'])
        model.eval()

        test_dataset = CustomDataset(mode=self.args.mode, flag='test')
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

        criterion = nn.MSELoss()
        total_loss = 0.0

        all_preds = []
        all_labels = []
        all_inps = []

        with torch.no_grad():
            loop = tqdm.tqdm(test_loader, desc='Test', unit='batch')
            for x, y in loop:
                x, y = x.to(self.device)[:, -self.args.input_len:, :], y.to(self.device)
                pred = model(x)
                loss = criterion(pred*12., y*12.)
                total_loss += loss.item() * x.size(0)
                loop.set_postfix(loss=loss.item())

                all_inps.append(x[:, :, -1].cpu())
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())

        avg_loss = total_loss / len(test_loader)
        print(f"Test loss: {avg_loss:.6f}")

        preds_tensor = torch.cat(all_preds, dim=0).numpy()
        labels_tensor = torch.cat(all_labels, dim=0).numpy()
        inps_tensor = torch.cat(all_inps, dim=0).numpy()

        save_path = os.path.join(self.args.res_dir, self.args.mode, self.args.ex_name, self.args.model_name)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'preds.npy'), preds_tensor)
        np.save(os.path.join(save_path, 'labels.npy'), labels_tensor)
        np.save(os.path.join(save_path, 'inps.npy'), inps_tensor)

        for i in range(24):
            mse = (labels_tensor[:, i]*12. - preds_tensor[:, i]*12.) ** 2.
            rmse = np.sqrt(mse.mean(axis=0))
            print(rmse)
