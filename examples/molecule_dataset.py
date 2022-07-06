import torch
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from rdkit.Chem import Draw
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import mean_squared_error as mse


from auglichem.molecule import Compose, RandomAtomMask, RandomBondDelete, MotifRemoval
from auglichem.molecule.data import MoleculeDatasetWrapper
from auglichem.molecule.models import GCN, AttentiveFP, GINE, DeepGCN

# %% md

### Set up dataset

# %%

# Create transformation
transform = Compose([
    RandomAtomMask([0.1, 0.3]),
    RandomBondDelete([0.1, 0.3]),
    # MotifRemoval()
])

# Initialize dataset object
dataset = MoleculeDatasetWrapper("SIDER", data_path="./data_download", transform=transform, batch_size=128)

# Get train/valid/test splits as loaders
train_loader, valid_loader, test_loader = dataset.get_data_loaders()

# %% md

### Initialize model with task from data

# %%

# Get model
model = DeepGCN(task=dataset.task)

# Uncomment the following line to use GPU
# model.cuda()

# %% md

### Initialize traning loop

# %%

if (dataset.task == 'classification'):
    criterion = torch.nn.CrossEntropyLoss()
elif (dataset.task == 'regression'):
    criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

if (dataset.task == 'classification'):
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)
elif (dataset.task == 'regression'):
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    criterion = torch.nn.MSELoss()


# %% md

### Train the model

def evaluate(model, test_loader, validation=False):
    task = test_loader.dataset.task
    set_str = "VALIDATION" if validation else "TEST"
    with torch.no_grad():
        model.eval()

        all_preds = torch.Tensor().cuda()
        all_labels = torch.Tensor().cuda()
        for data in test_loader:
            # _, pred = model(data)

            # data -> GPU
            _, pred = model(data.cuda())

            # Hold on to all predictions and labels
            if (task == 'classification'):
                # all_preds.extend(pred[:,1])
                all_preds = torch.cat([all_preds, pred[:, 1]])
            elif (task == 'regression'):
                # all_preds.extend(pred)
                all_preds = torch.cat([all_preds, pred])

            # all_labels.extend(data.y)
            all_labels = torch.cat([all_labels, data.y])

        if (task == 'classification'):
            metric = ras(all_labels.cpu(), all_preds.cpu().detach())
            print("{0} ROC: {1:.3f}".format(set_str, metric))
        elif (task == 'regression'):
            metric = mse(all_labels.cpu(), all_preds.cpu().detach(), squared=False)
            print("{0} RMSE: {1:.3f}".format(set_str, metric))
        return metric

# %%

best_metric = -1 if train_loader.dataset.task == "classification" else float('inf')
best_test = -1 if train_loader.dataset.task == "classification" else float('inf')
early_stop = 0



for epoch in range(100):
    for bn, data in tqdm(enumerate(train_loader)):

        optimizer.zero_grad()

        _, pred = model(data)

        # data -> GPU

        if (train_loader.dataset.task == "classification"):
            loss = criterion(pred, data.y.flatten())
        if (train_loader.dataset.task == "regression"):
            loss = criterion(pred[:, 0], data.y.flatten())
        loss.backward()
        optimizer.step()

    val_metric = evaluate(model, valid_loader, True)
    if (train_loader.dataset.task == "classification"):
        if val_metric > best_metric:
            early_stop = 0
            best_metric = val_metric
            test_metric = evaluate(model, test_loader)
            best_test = test_metric
            print("best update:" + str(test_metric))
    if (train_loader.dataset.task == "regression"):
        if val_metric < best_metric:
            early_stop = 0
            best_metric = val_metric
            test_metric = evaluate(model, test_loader)
            best_test = test_metric
            print("best update:" + str(test_metric))

    early_stop += 1
    scheduler.step(val_metric)
    if early_stop == 20:
        break

print("best test:" + str(best_test))

#