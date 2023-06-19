from chemicalx.models import DeepDrug, CASTER, SSIDDI, DeepDDI, EPGCNDS, GCNBMP, MRGNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc
from chemicalx.loss import CASTERSupervisedLoss
from chemicalx import pipeline
from contrast_DatasetLoader import myDatasetLoader
from sklearn.metrics import roc_auc_score
import os
import copy
import argparse
import numpy as np
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print(torch.cuda.current_device())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcnbmp',
                        help='model select to train')
    parser.add_argument('--kfold', type=int, default=5,
                        help='kfold')
    return parser.parse_args()


def train_caster(dataset):
    model = CASTER(drug_channels=dataset.drug_channels)
    results = pipeline(
        dataset=dataset,
        model=model,
        loss_cls=CASTERSupervisedLoss,
        batch_size=1024,
        train_size=0.8,
        epochs=100,
        context_features=False,
        drug_features=True,
        drug_molecules=False,
    )
    return results


def train_deepdrug(dataset):
    model = DeepDrug()
    results = pipeline(
        dataset=dataset,
        model=model,
        train_size=0.8,
        optimizer_kwargs=dict(lr=0.001),
        batch_size=1024,
        epochs=20,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    return results


def train_ssiddi(dataset):
    model = SSIDDI()
    results = pipeline(
        dataset=dataset,
        model=model,
        train_size=0.8,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10**-7),
        batch_size=1024,
        epochs=20,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    return results


def train_deepddi(dataset):
    model = DeepDDI(drug_channels=dataset.drug_channels, hidden_layers_num=2)   # drug_channel就是每个药物的维度 300
    results = pipeline(
        dataset=dataset,
        model=model,
        train_size=0.8,
        batch_size=1024,
        epochs=20,
        context_features=False,
        drug_features=True,
        drug_molecules=False,
    )
    return results


def train_epgcnds(dataset):
    model = EPGCNDS()
    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10**-7),
        batch_size=1024,
        epochs=20,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    return results


def train_gcnbmp(dataset):
    model = GCNBMP(hidden_conv_layers=2)

    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10**-7),
        batch_size=1024,
        epochs=20,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    return results


def train_mrgnn(dataset):
    model = MRGNN()
    results = pipeline(
        dataset=dataset,
        model=model,
        optimizer_kwargs=dict(lr=0.01, weight_decay=10**-7),
        batch_size=1024,
        epochs=20,
        context_features=False,
        drug_features=True,
        drug_molecules=True,
    )
    return results


def calc_metrics(predictions_df):
    prediction = copy.deepcopy(predictions_df['prediction'])
    prediction = prediction.to_list()
    prediction = [1 if x > 0.5 else 0 for x in prediction]
    prediction = np.array(prediction)
    label = copy.deepcopy(predictions_df['label'])
    label = np.array(label, dtype=int)

    precision = precision_score(label, prediction)
    recall = recall_score(label, prediction)
    f1 = f1_score(label, prediction)
    acc = accuracy_score(label, prediction)
    roc_auc = roc_auc_score(predictions_df["label"], predictions_df["prediction"])
    aupr_precision, aupr_recall, _ = precision_recall_curve(predictions_df["label"], predictions_df["prediction"])
    aupr = auc(aupr_recall, aupr_precision)
    return precision, recall, f1, acc, roc_auc, aupr


def main():
    args = parse_args()
    model_mode = args.model
    print(model_mode)
    kfold = args.kfold
    data_path = '../data_after_processing/contrast_test'
    dataset = myDatasetLoader(data_path)
    aupr_list = []
    auroc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    acc_list = []
    for i in range(kfold):
        if model_mode == 'caster':
            results = train_caster(dataset)
        elif model_mode == 'deepddi':
            results = train_deepddi(dataset)
        elif model_mode == 'ssiddi':
            results = train_ssiddi(dataset)
        elif model_mode == 'deepdrug':
            results = train_deepdrug(dataset)
        elif model_mode == 'epgcnds':
            results = train_epgcnds(dataset)
        elif model_mode == 'gcnbmp':
            results = train_gcnbmp(dataset)
        elif model_mode == 'mrgnn':
            results = train_mrgnn(dataset)

        predictions_df = results.predictions
        precision, recall, f1, acc, roc_auc, aupr = calc_metrics(predictions_df)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        acc_list.append(acc)
        auroc_list.append(roc_auc)
        aupr_list.append(aupr)
    print('KFold_precision:', np.mean(precision_list))
    print('KFold_precision vearance:', np.var(precision_list))
    print('KFold_recall:', np.mean(recall_list))
    print('KFold_recall vearance:', np.var(recall_list))
    print('KFold_acc:', np.mean(acc_list))
    print('KFold_acc vearance:', np.var(acc_list))
    print('KFold F1:', np.mean(f1_list))
    print('KFold F1 vearance:', np.var(f1_list))
    print('KFold ROC-AUC:', np.mean(auroc_list))
    print('KFold ROC-AUC vearance:', np.var(auroc_list))
    print('KFold PR-AUC', np.mean(aupr_list))
    print('KFold PR-AUC vearance:', np.var(aupr_list))


if __name__ == "__main__":
    main()
