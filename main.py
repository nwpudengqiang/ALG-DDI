import numpy as np
import torch
import torch.optim as optim
import os
import random
import torch.utils.data as Data

from utils import parse_args, focal_loss, evaluate, early_stopping
from DataLoader import DataLoader
from model import My_Model
from plt import draw_auc, draw_aupr

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# --------------------------------------------- main -------------------------------------------------------------------
args = parse_args()
# seed设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print('多分类 {} | ddi数据源 {} | 尺度信息 {} | 融合策略 {}'
      .format(args.multi_class, args.ddi_dataset, args.feature_source, args.fusion_type))
# GPU / CPU选择使用GPU还是CPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()  # 1
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

# 初始化数据  
data = DataLoader(args)
# 将数据转化为tensor，这样可以在gpu上运行
structure_pre_embed = torch.tensor(data.structure_embed).float().to(device)
drug_protein_embed = torch.tensor(data.drug_protein_embed).float().to(device)
drug_effect_embed = torch.tensor(data.drug_effect_embed).float().to(device)
drug_disease_embed = torch.tensor(data.drug_disease_embed).float().to(device)
kg_pre_embed = torch.tensor(data.kg_embed).float().to(device)
pre_embedding = [structure_pre_embed, drug_protein_embed, drug_effect_embed, drug_disease_embed, kg_pre_embed]

# 用于储存五折交叉验证得到的五次结果  下面是二分类
all_acc_list, all_precision_list, all_recall_list, all_f1_list, all_auroc_list, all_aupr_list = [], [], [], [], [], []
all_micro_acc_list, all_micro_precision_list, all_micro_recall_list, all_micro_f1_list = [], [], [], []
all_prediction_list, all_label_list = [], []        # 用来储存每个fold得到的预测值和真实值，用于画图
all_model_list = []

for i in range(args.n_fold):
    fold = i + 1
    print('begin the {} fold'.format(fold))
    # Data.TensorDataset()里的两个输入是tensor类型，所以需要变成张量
    train_x = torch.tensor(data.DDI_train_data_X[i])
    train_y = torch.tensor(data.DDI_train_data_Y[i])
    test_x = torch.tensor(data.DDI_test_data_X[i])
    test_y = torch.tensor(data.DDI_test_data_Y[i])
    # 数据封装利用的是TensorDataset，将数据和label封装在一起
    torch_dataset_train = Data.TensorDataset(train_x, train_y)
    torch_dataset_test = Data.TensorDataset(test_x, test_y)
    # 将数据划分为不同的batch
    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=args.DDI_batch_size,
        shuffle=True)
    # if args.multi_class:
    #     args.DDI_evaluate_size = data.n_ddi_test[i]
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=args.DDI_evaluate_size,
        shuffle=True)

    # 定义模型
    model = My_Model(args)
    model.to(device)
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # lr是learning rate

    epoch_list = []
    # 用于记录每一次评估得到指标，如果每个epoch评估一次，那么每个epoch就会添加一次
    precision_list, recall_list, f1_list, auroc_list, acc_list, aupr_list = [], [], [], [], [], []
    micro_precision_list, micro_recall_list, micro_f1_list, micro_acc_list = [], [], [], []
    prediction_list, label_list = [], []

    for epoch in range(1, args.n_epoch + 1):
        if args.multi_class:
            if epoch <= args.n_epoch // 2:
                loss_func = torch.nn.CrossEntropyLoss()
            else:
                loss_func = focal_loss()
        else:
            loss_func = torch.nn.BCEWithLogitsLoss()
        model.train()  # 每一个epoch训练一次
        ddi_total_loss = 0
        n_ddi_batch = data.n_ddi_train[i] // args.DDI_batch_size + 1  # 计算一共需要多少个batch

        for step, (batch_x, batch_y) in enumerate(loader_train):  # 遍历
            iter = step + 1
            if use_cuda:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out = model('calc_ddi_loss', batch_x, pre_embedding)
            if not args.multi_class:
                out = out.squeeze(-1)       # 将n*1的张量变成1*n
                loss = loss_func(out, batch_y.float())
            else:
                loss = loss_func(out, batch_y.long())
            loss.backward()
            optimizer.step()
            ddi_total_loss += loss.item()
        # 每次epoch输出一次结果
        print('DDI Training: Epoch {:03d}/{:03d} | Total Iter {:04d} | Iter Mean Loss {:.4f}'
              .format(epoch, args.n_epoch, n_ddi_batch, ddi_total_loss / n_ddi_batch))

        if (epoch % args.evaluate_every) == 0:
            if not args.multi_class:
                precision, recall, f1, acc, auroc, aupr, prediction, label = evaluate(args, model, loader_test, pre_embedding)
                epoch_list.append(epoch)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                acc_list.append(acc)
                auroc_list.append(auroc)
                aupr_list.append(aupr)
                prediction_list.append(prediction)
                label_list.append(label)
                best_auroc, should_stop = early_stopping(auroc_list, args.stopping_steps)  # 早停法解决过拟合，这里是以auroc作为主要指标

                if should_stop:  # 当出现过拟合时，需要跳出循环
                    index = auroc_list.index(best_auroc)
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])
                    all_acc_list.append(acc_list[index])
                    all_auroc_list.append(auroc_list[index])
                    all_aupr_list.append(aupr_list[index])
                    all_prediction_list.append(prediction_list[index])
                    all_label_list.append(label_list[index])
                    print("出现过拟合现象，后面的epoch不再进行")
                    print('fold {:01d} : precision {:.4f} recall {:.4f} f1 {:.4f} acc {:.4f} auroc {:.4f} aupr {:.4f}'
                          .format((i + 1), precision_list[index], recall_list[index], f1_list[index], acc_list[index],
                                  auroc_list[index], aupr_list[index]))
                    all_model_list.append(model)
                    break

                if epoch == args.n_epoch:  # 当运行到最后一个epoch时结束循环
                    index = auroc_list.index(best_auroc)
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])
                    all_acc_list.append(acc_list[index])
                    all_auroc_list.append(auroc_list[index])
                    all_aupr_list.append(aupr_list[index])
                    all_prediction_list.append(prediction_list[index])
                    all_label_list.append(label_list[index])
                    print('未出现过拟合')
                    print('fold {:01d} : precision {:.4f} recall {:.4f} f1 {:.4f} acc {:.4f} auroc {:.4f} aupr {:.4f}'
                          .format((i + 1), precision_list[index], recall_list[index], f1_list[index], acc_list[index],
                                  auroc_list[index], aupr_list[index]))
                    all_model_list.append(model)
            else:
                micro_precision, micro_recall, micro_f1, micro_acc = evaluate(args, model, loader_test, pre_embedding)
                micro_precision_list.append(micro_precision)
                micro_recall_list.append(micro_recall)
                micro_f1_list.append(micro_f1)
                micro_acc_list.append(micro_acc)
                best_f1, should_stop = early_stopping(micro_f1_list, args.stopping_steps)
                # if should_stop:  # 当出现过拟合时，需要跳出循环
                #     index = micro_f1_list.index(best_f1)
                #     all_micro_precision_list.append(micro_precision_list[index])
                #     all_micro_recall_list.append(micro_recall_list[index])
                #     all_micro_f1_list.append(micro_f1_list[index])
                #     all_micro_acc_list.append(micro_acc_list[index])
                #     print("出现过拟合现象，后面的epoch不再进行")
                #     print('fold {:01d} : micro_precision {:.4f} micro_recall {:.4f} micro_f1 {:.4f} acc {:.4f} '
                #           .format((i + 1), micro_precision_list[index], micro_recall_list[index], micro_f1_list[index],
                #                   micro_acc_list[index]))
                #     break

                if epoch == args.n_epoch:  # 当运行到最后一个epoch时结束循环
                    index = micro_f1_list.index(best_f1)
                    all_micro_precision_list.append(micro_precision_list[index])
                    all_micro_recall_list.append(micro_recall_list[index])
                    all_micro_f1_list.append(micro_f1_list[index])
                    all_micro_acc_list.append(micro_acc_list[index])
                    print('未出现过拟合')
                    print('fold {:01d} : micro_precision {:.4f} micro_recall {:.4f} micro_f1 {:.4f} acc {:.4f} '
                          .format((i + 1), micro_precision_list[index], micro_recall_list[index], micro_f1_list[index],
                                  micro_acc_list[index]))

if not args.multi_class:
    mean_precision = np.mean(all_precision_list)
    precision_var = np.var(all_precision_list)
    mean_recall = np.mean(all_recall_list)
    recall_var = np.var(all_recall_list)
    mean_f1 = np.mean(all_f1_list)
    f1_var = np.var(all_f1_list)
    mean_acc = np.mean(all_acc_list)
    acc_var = np.var(all_acc_list)
    mean_auroc = np.mean(all_auroc_list)
    auroc_var = np.var(all_auroc_list)
    mean_aupr = np.mean(all_aupr_list)
    aupr_var = np.var(all_aupr_list)

    print('{}折平均值 : precision {:.4f} recall {:.4f} f1 {:.4f} acc {:.4f} auroc {:.4f} aupr {:.4f}'
          .format(args.n_fold, mean_precision, mean_recall, mean_f1, mean_acc, mean_auroc, mean_aupr))
    print('对应标准差 : precision {:.4f} recall {:.4f} f1 {:.4f} acc {:.4f} auroc {:.4f} aupr {:.4f}'
          .format(args.n_fold, precision_var, recall_var, f1_var, acc_var, auroc_var, aupr_var))
    max_auroc = max(all_auroc_list)  # 找到列表中的最大值
    max_index = all_auroc_list.index(max_auroc)
    final_model = all_model_list[max_index]
    torch.save(final_model, '{}_model.pth'.format(args.case_drug))
else:
    mean_micro_precision = np.mean(all_micro_precision_list)
    mean_micro_recall = np.mean(all_micro_recall_list)
    mean_micro_f1 = np.mean(all_micro_f1_list)
    mean_micro_acc = np.mean(all_micro_acc_list)
    print('{}折平均值 : micro_precision {:.4f} micro_recall {:.4f} micro_f1 {:.4f} acc {:.4f}'
          .format(args.n_fold, mean_micro_precision, mean_micro_recall, mean_micro_f1, mean_micro_acc))

if args.plt:
    auc_curve_file = 'plt/{}_roc_curve.png'.format(args.ddi_dataset)
    draw_auc(all_label_list, all_prediction_list, auc_curve_file)
    aupr_curve_file = 'plt/{}_aupr_curve.png'.format(args.ddi_dataset)
    draw_aupr(all_label_list, all_prediction_list, aupr_curve_file)
