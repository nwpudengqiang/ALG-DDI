import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import copy
import numpy as np


# initial param
def parse_args():
    parser = argparse.ArgumentParser(description="Run code.")
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed.')

    # 使用哪些数据以及使用什么融合策略
    parser.add_argument('--original_kg', type=str, default='primeKG',
                        choices=['primeKG', 'drkg'],
                        help='the source of kg')
    parser.add_argument('--feature_source', type=str, default='all',
                        help='0:only use structure information '
                             '1:only use protein-drug information '
                             '2:only use effect-drug information '
                             '3:only use disease-drug information '
                             '4:only use kg information '
                             '04:use st and kg information'
                             'all：use all of them')
    parser.add_argument('--KGE_model', type=str, default='ComplEx',
                        choices=['TransE_l1', 'TransE_l2', 'TransR', 'RESCAL', 'DistMult', 'ComplEx', 'RotatE', 'SimplE'],
                        help='the model of KGE')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'mean', 'DNN', 'element-wise', 'attention'],
                        help='fusion strategy')
    parser.add_argument('--multi_class', type=bool, default=False,
                        help='Binary-class or multi-class')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='FC output dim: 1(for binary-class) or 82(for multi class)')
    parser.add_argument('--ddi_dataset', type=str, default='DS1_without_case_drug',
                        choices=['DS1', 'DS2', 'DS3', 'DS1_without_case_drug'],
                        help='DS1:from_kg, DS2:drugbank_v5 DS3:drugbank_v4 DS1_without_case_drug:removed case drug')

    # 一些超参数
    parser.add_argument('--DDI_batch_size', type=int, default=2048,
                        help='DDI batch size.')
    parser.add_argument('--DDI_evaluate_size', type=int, default=2500,
                        help='KG batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--n_fold', type=int, default=5,
                        help='number of fold')
    parser.add_argument('--n_epoch', type=int, default=100,  # 原文章是200
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,  # 原文是10,代表如果连续十次评估性能都在下降，则出现了过拟合
                        help='Number of epoch for early stopping')
    parser.add_argument('--evaluate_every', type=int, default=1,  # 代表每一次epoch就评估一次模型
                        help='Epoch interval of evaluating DDI.')

    # 如果使用attention作为融合策略，需要使用的相关参数
    parser.add_argument('--attention_hidden', type=int, default=128,
                        help='the dim of hidden layer in attention')
    parser.add_argument('--attention_out', type=int, default=1,
                        help='attention out dim')

    # 预测器网络中的相关参数
    parser.add_argument('--feature_dim', type=int, default='300',
                        help='Dimension of single source')
    parser.add_argument('--n_hidden_1', type=int, default=256,
                        help='FC hidden 1 dim')
    parser.add_argument('--n_hidden_2', type=int, default=128,
                        help='FC hidden 2 dim')

    parser.add_argument('--plt', type=bool, default=False,
                        help='draw charts or not')
    parser.add_argument('--case_drug', type=str, default='Cannabidiol',
                        choices=['Cannabidiol', 'Cisplatin', 'Dexamethasone'],
                        help='Drugs used for case study')
    args = parser.parse_args()

    return args


class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss, self).__init__()

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim() == 2 and labels.dim()==1
        labels = labels.view(-1, 1)  # 变成列向量[B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]
        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax
        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss


def calc_metrics(y_true, y_pred, pred_score, multi_class):
    if not multi_class:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true.cuda().data.cpu().numpy(), pred_score.cuda().data.cpu().numpy())
        aupr_precision, aupr_recall, _ = precision_recall_curve(y_true.cuda().data.cpu().numpy(), pred_score.cuda().data.cpu().numpy())
        aupr = auc(aupr_recall, aupr_precision)
        return precision, recall, f1, acc, auroc, aupr
    else:
        micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        acc = accuracy_score(y_true, y_pred)
        # print(micro_precision, micro_recall, micro_f1, acc)
        return micro_precision, micro_recall, micro_f1, acc


def evaluate(args, model, loader_test, pre_embedding):
    model.eval()
    if not args.multi_class:
        precision_list = []
        recall_list = []
        f1_list = []
        acc_list = []
        auroc_list = []
        aupr_list = []
        all_prediction = []
        all_true = []
        with torch.no_grad():
            for data in loader_test:
                test_x, test_y = data
                out = model('predict', test_x, pre_embedding)
                out = out.squeeze(-1)
                prediction = copy.deepcopy(out)
                prediction[prediction >= 0.5] = 1
                prediction[prediction < 0.5] = 0
                prediction = prediction.cuda().data.cpu().numpy()
                # 每执行一次evaluate函数，就会执行很多次calc_metrics
                precision, recall, f1, acc, auroc, aupr = calc_metrics(test_y, prediction, out, args.multi_class)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                acc_list.append(acc)
                auroc_list.append(auroc)
                aupr_list.append(aupr)

                batch_prediction = out.cuda().data.cpu().numpy()
                all_prediction.append(batch_prediction)
                batch_true = test_y.numpy()
                all_true.append(batch_true)
        # 对每个batch得到的指标进行平均
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1 = np.mean(f1_list)
        acc = np.mean(acc_list)
        auroc = np.mean(auroc_list)
        aupr = np.mean(aupr_list)
        all_prediction = np.concatenate(all_prediction)
        all_true = np.concatenate(all_true)
        return precision, recall, f1, acc, auroc, aupr, all_prediction, all_true

    else:
        micro_precision_list = []
        micro_recall_list = []
        micro_f1_list = []
        acc_list = []
        with torch.no_grad():
            for data in loader_test:
                test_x, test_y = data
                out = model('predict', test_x, pre_embedding)
                prediction = torch.max(out, 1)[1]
                prediction = prediction.cuda().data.cpu().numpy()
                micro_precision, micro_recall, micro_f1, acc = calc_metrics(test_y, prediction, out, args.multi_class)
                micro_precision_list.append(micro_precision)
                micro_recall_list.append(micro_recall)
                micro_f1_list.append(micro_f1)
                acc_list.append(acc)

        micro_precision = np.mean(micro_precision_list)
        micro_recall = np.mean(micro_recall_list)
        micro_f1 = np.mean(micro_f1_list)
        acc = np.mean(acc_list)
        return micro_precision, micro_recall, micro_f1, acc


def early_stopping(metric_list, stopping_steps):
    best_metric = max(metric_list)
    best_step = metric_list.index(best_metric)
    # 如果连续多次epoch得到的指标都在下降，那么应该停止了
    if len(metric_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_metric, should_stop