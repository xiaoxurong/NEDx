import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import binary_classification_metrics, multiclass_classification_metrics
from utils.tools import EarlyStopping, adjust_learning_rate

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
        outputs = self.model(batch_x)          # (B, 1), (B, D), or (B, T, D)
        if outputs.dim() == 3:
            outputs = outputs.mean(dim=1)      # global avg pool over time → (B, D)
        return outputs

    # ─────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        val_data,   val_loader   = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer     = self._select_optimizer()
        criterion     = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.forward_classification(batch_x)

                if self.args.num_classes == 2:
                    loss = criterion(logits.squeeze(-1), batch_y.float())
                else:
                    loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_loss, val_metric, _, _ = self.validate(val_loader, criterion)

            print(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss {np.mean(train_losses):.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"AUROC {val_metric['auroc']:.4f} | "
                f"F1 {val_metric['f1']:.4f}"
            )

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Restore best checkpoint
        best_ckpt = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        return self.model

    # ─────────────────────────────────────────────────────────────────────────
    # Test  — runs on the held-out test set and returns a metrics dict
    # ─────────────────────────────────────────────────────────────────────────

    def test(self, setting) -> dict:
        """
        Evaluate on the test split using the best checkpoint saved during train().
        Returns a dict of metrics (auroc, f1, acc, …) for fold aggregation in run.py.
        """
        test_data, test_loader = self._get_data('test')

        # Best checkpoint was already loaded at the end of train(), but reload
        # explicitly here so test() can also be called standalone.
        ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        else:
            print(f"Warning: checkpoint not found at {ckpt_path}, using current model weights.")

        criterion = self._select_criterion()
        test_loss, test_metric, test_preds, test_trues = self.validate(test_loader, criterion)

        print(
            f"[Test]  Loss {test_loss:.4f} | "
            + " | ".join(f"{k.upper()} {v:.4f}" for k, v in test_metric.items())
        )

        return test_metric   # e.g. {'auroc': 0.72, 'f1': 0.45, 'acc': 0.88, ...}

    # ─────────────────────────────────────────────────────────────────────────
    # Validate  — shared by train loop and test()
    # ─────────────────────────────────────────────────────────────────────────

    def validate(self, loader, criterion):
        self.model.eval()
        losses, preds, trues = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.forward_classification(batch_x)

                if self.args.num_classes == 2:
                    logits = logits.squeeze(-1)                       # (B,)
                    loss   = criterion(logits, batch_y.float())
                    prob   = torch.sigmoid(logits).cpu().numpy()
                else:
                    loss = criterion(logits, batch_y)
                    prob = torch.softmax(logits, dim=1).cpu().numpy()

                losses.append(loss.item())
                preds.append(prob)
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        # Only warn when something is actually wrong — don't print every epoch
        if np.isnan(preds).any() or np.isinf(preds).any():
            print(f"WARNING: predictions contain NaN/Inf! "
                  f"min={np.nanmin(preds):.4f}  max={np.nanmax(preds):.4f}")

        if self.args.num_classes == 2:
            metric = binary_classification_metrics(trues, preds, threshold=0.5)
        else:
            metric = multiclass_classification_metrics(trues, preds, average="macro")

        return np.mean(losses), metric, preds, trues