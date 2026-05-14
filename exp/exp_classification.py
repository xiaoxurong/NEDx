import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import *
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.args.num_classes == 2:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def forward_classification(self, batch_x):
        outputs = self.model(batch_x)   # (B, T, D) or (B, D)

        if outputs.dim() == 3:
            outputs = outputs.mean(dim=1)  # global average pooling

        return outputs  # (B, num_classes)
    
    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.forward_classification(batch_x)

                if self.args.num_classes == 2:
                    batch_y = batch_y.float()
                    loss = criterion(logits.squeeze(-1), batch_y)
                else:
                    loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            val_loss, val_metric, val_preds, val_trues = self.validate(val_loader, criterion)

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss {np.mean(train_loss):.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"AUROC {val_metric['auroc']:.4f} | "
                f"F1 {val_metric['f1']:.4f}"
            )

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                break

        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        return self.model

    def validate(self, loader, criterion):
        self.model.eval()
        losses, preds, trues = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.forward_classification(batch_x)

                if self.args.num_classes == 2:
                    logits = logits.squeeze(-1)   # enforce (B,)
                    loss = criterion(logits, batch_y.float())
                    prob = torch.sigmoid(logits).cpu().numpy()
                else:
                    loss = criterion(logits, batch_y)
                    prob = torch.softmax(logits, dim=1).cpu().numpy()

                losses.append(loss.item())
                preds.append(prob)
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds)
        print("Any NaN in preds?", np.isnan(preds).any())
        print("Any inf in preds?", np.isinf(preds).any())
        print("Pred stats:", np.nanmin(preds), np.nanmax(preds))
        trues = np.concatenate(trues)

        if self.args.num_classes == 2:
            metric = binary_classification_metrics(trues, preds, threshold=0.5)
        else:
            metric = multiclass_classification_metrics(trues, preds, average="macro")

        return np.mean(losses), metric, preds, trues