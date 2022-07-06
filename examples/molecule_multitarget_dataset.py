
import torch
from tqdm import tqdm
import numpy as np


from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import mean_squared_error


from auglichem.molecule import Compose, RandomAtomMask, RandomBondDelete, MotifRemoval
from auglichem.molecule.data import MoleculeDatasetWrapper
from auglichem.molecule.models import GCN, AttentiveFP, GINE, DeepGCN

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Create transformation
transform = Compose([
    RandomAtomMask([0.1, 0.3]),
    RandomBondDelete([0.1, 0.3]),
    # MotifRemoval()
])
# transform = RandomAtomMask(0.1)

# Initialize dataset object
dataset = MoleculeDatasetWrapper("SIDER", data_path="./data_download", transform=transform, batch_size=128)

# Get train/valid/test splits as loaders
train_loader, valid_loader, test_loader = dataset.get_data_loaders("all")


### Initialize model with task from data


# Get model
num_outputs = len(dataset.labels.keys())
model = GCN(task=dataset.task, output_dim=num_outputs)

# Uncomment the following line to use GPU
model.cuda()


### Initialize traning loop


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

best_metric = -1 if train_loader.dataset.task == "classification" else float('inf')
best_test = -1 if train_loader.dataset.task == "classification" else float('inf')
early_stop = 0

### Test the model

def evaluate(model, test_loader, validation=False):
    set_str = "VALIDATION" if validation else "TEST"
    with torch.no_grad():

        # All targets we're evaluating
        target_list = test_loader.dataset.target

        # Dictionaries to keep track of predictions and labels for all targets
        all_preds = {target: [] for target in target_list}
        all_labels = {target: [] for target in target_list}

        model.eval()
        for data in test_loader:
            # Get prediction for all data
            # _, pred = model(data)

            # To use GPU, data must be cast to cuda
            _, pred = model(data.cuda())

            for idx, target in enumerate(target_list):
                # Get indices where target has a value
                good_idx = np.where(data.y[:, idx].cpu() != -999999999)

                # When the data is placed on GPU, target must come back to CPU
                # good_idx = np.where(data.y.cpu()[:,idx]!=-999999999)

                # Prediction is handled differently for classification and regression
                if (train_loader.dataset.task == 'classification'):
                    current_preds = pred[:, 2 * (idx):2 * (idx + 1)][good_idx][:, 1]
                    current_labels = data.y[:, idx][good_idx]
                elif (train_loader.dataset.task == 'regression'):
                    current_preds = pred[:, idx][good_idx]
                    current_labels = data.y[:, idx][good_idx]

                # Save predictions and targets
                all_preds[target].extend(list(current_preds.detach().cpu().numpy()))
                all_labels[target].extend(list(current_labels.detach().cpu().numpy()))

        scores = {target: None for target in target_list}
        for target in target_list:
            if (test_loader.dataset.task == 'classification'):
                scores[target] = ras(all_labels[target], all_preds[target])
                print("{0} {1} ROC: {2:.5f}".format(target, set_str, scores[target]))
            elif (test_loader.dataset.task == 'regression'):
                scores[target] = mean_squared_error(all_labels[target], all_preds[target],
                                                    squared=False)
                print("{0} {1} RMSE: {2:.5f}".format(target, set_str, scores[target]))
        print(scores)
        return mean(Dict.values())

### Train the model


for epoch in range(100):
    for bn, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        loss = 0.

        # Get prediction for all data
        # _, pred = model(data)

        # To use GPU, data must be cast to cuda
        _, pred = model(data.cuda())

        for idx, t in enumerate(train_loader.dataset.target):
            # Get indices where target has a value
            good_idx = np.where(data.y[:, idx].cpu() != -999999999)

            # When the data is placed on GPU, target must come back to CPU
            # good_idx = np.where(data.y.cpu()[:,idx]!=-999999999)

            # Prediction is handled differently for classification and regression
            if (train_loader.dataset.task == 'classification'):
                current_preds = pred[:, 2 * (idx):2 * (idx + 1)][good_idx]
                current_labels = data.y[:, idx][good_idx]
            elif (train_loader.dataset.task == 'regression'):
                current_preds = pred[:, idx][good_idx]
                current_labels = data.y[:, idx][good_idx]

            loss += criterion(current_preds, current_labels)

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



