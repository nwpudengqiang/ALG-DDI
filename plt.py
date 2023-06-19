import networkx as nx
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def draw_network(edge_list):
    # 创建一个空的无向图
    G = nx.Graph()
    # 从边列表中添加边到图中
    # edge_list = [(0, 1), (1, 2), (2, 3), (3, 0)]
    G.add_edges_from(edge_list)

    # 绘制网络图
    nx.draw(G, with_labels=True)

    # 显示图形
    plt.show()


def draw_auc(predictions, labels, store_file):
    fig, ax = plt.subplots()
    fpr_list, tpr_list, roc_auc_list = [], [], []
    for i in range(len(predictions)):
        y_true = predictions[i]
        y_scores = labels[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)      # 计算真正例率和假正例率
        roc_auc = auc(fpr, tpr)     # 计算曲线下的面积
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        ax.plot(fpr, tpr, label='fold{} AUC = {:.4f}'.format(i, roc_auc))

    # 设置图例和标题
    ax.legend(loc='lower right')
    ax.set_title('ROC curve mean-AUC={:.4f}'.format(np.mean(roc_auc_list)))
    # 放大左上角区域
    axins = ax.inset_axes([0.6, 0.4, 0.35, 0.35])  # 设置放大区域的位置和大小
    for i in range(len(fpr_list)):
        axins.plot(fpr_list[i], tpr_list[i], label='fold {} (AUC = {:.4f})'.format(i, roc_auc_list[i]))
    axins.set_xlim(0, 0.2)  # 设置放大区域的 x 轴范围
    axins.set_ylim(0.8, 1)  # 设置放大区域的 y 轴范围
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines['top'].set_linestyle('--')
    axins.spines['bottom'].set_linestyle('--')
    axins.spines['left'].set_linestyle('--')
    axins.spines['right'].set_linestyle('--')
    border_color = 'lightgray'
    axins.spines['top'].set_color(border_color)
    axins.spines['bottom'].set_color(border_color)
    axins.spines['left'].set_color(border_color)
    axins.spines['right'].set_color(border_color)
    ax.indicate_inset_zoom(axins)  # 在原图上标示放大区域
    plt.savefig(store_file, dpi=300)


def draw_aupr(predictions, labels, store_file):
    fig, ax = plt.subplots()
    precision_list, recall_list, pr_auc_list = [], [], []
    for i in range(len(predictions)):
        y_true = predictions[i]
        y_scores = labels[i]
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        precision_list.append(precision)
        recall_list.append(recall)
        pr_auc_list.append(pr_auc)
        ax.plot(recall, precision, label='fold{} AUC = {:.4f}'.format(i, pr_auc))

    # 设置图例和标题
    ax.legend(loc='lower left')
    ax.set_title('PR curve mean-AUC={:.4f}'.format(np.mean(pr_auc_list)))
    # 放大左上角区域
    axins = ax.inset_axes([0.45, 0.02, 0.25, 0.35])  # 设置放大区域的位置和大小
    for i in range(len(precision_list)):
        axins.plot(precision_list[i], recall_list[i], label='fold {} (AUC = {:.4f})'.format(i, pr_auc_list[i]))
    axins.set_xlim(0.8, 1)  # 设置放大区域的 x 轴范围
    axins.set_ylim(0.8, 1)  # 设置放大区域的 y 轴范围
    axins.set_xticks([])
    axins.set_yticks([])
    axins.spines['top'].set_linestyle('--')
    axins.spines['bottom'].set_linestyle('--')
    axins.spines['left'].set_linestyle('--')
    axins.spines['right'].set_linestyle('--')
    border_color = 'lightgray'
    axins.spines['top'].set_color(border_color)
    axins.spines['bottom'].set_color(border_color)
    axins.spines['left'].set_color(border_color)
    axins.spines['right'].set_color(border_color)
    ax.indicate_inset_zoom(axins)  # 在原图上标示放大区域
    plt.savefig(store_file, dpi=300)

# ----------------------------------------------plt---------------------------------------------------------------------
# def save_loss(list, fold):
#     train_loss_file = 'plt/{}_fold{}_loss'.format(args.feature_source, fold)
#     with open(train_loss_file, 'w') as loss_file:
#         loss_file.writelines(list)
#
#
# def distribution_map(tensor):
#     plt.hist(tensor, bins=10)
#     plt.xlabel('predicted result')
#     plt.ylabel('frequency')
#     plt.show()
