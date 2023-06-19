import pandas as pd
import torch
import csv
import numpy as np
import torch.nn as nn
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)


def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id


def built_heterograph(all_drug_features, all_drug: dict, related_drug: dict, p_e_d_dict: dict, p_e_d: str, edges: list):
    # 首先构建异构图的边
    data = HeteroData()
    left = [edge[0] for edge in edges]
    right = [edge[1] for edge in edges]
    edge_index = torch.tensor([left, right])
    # 然后定义异构图节点的特征，all_drug的特征是One-hot向量，在某种局部信息层面，就取出其中包含的drug的特征， p_e_d是随机初始化向量
    num_related_drug, num_p_e_d = len(related_drug), len(p_e_d_dict)
    features = []
    for drug in related_drug.keys():
        id_drug = all_drug[drug]
        feature = all_drug_features[id_drug].tolist()
        features.append(feature)

    drug_feats = torch.tensor(features)  # 6282*300
    p_e_d_features = torch.randn(num_p_e_d, 300)  # 3094*300
    data['drug'].x = drug_feats
    data[p_e_d].x = p_e_d_features
    data['drug', 'with', p_e_d].edge_index = edge_index

    return data


class GNNEncoder(nn.Module):
    def __init__(self, hidden_channel_1, hidden_channel_2):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channel_1)
        self.conv2 = SAGEConv((-1, -1), hidden_channel_2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index, p_e_d):
        row, col = edge_label_index
        z = torch.cat([z_dict['drug'][row], z_dict[p_e_d][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2, data):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels_1, hidden_channels_2)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')  # 将编码器包装成一个异质图编码器
        self.decoder = EdgeDecoder(hidden_channels_2)

    def forward(self, x_dict, edge_index_dict, edge_label_index, p_e_d):
        # 输入drug和protein的特征x_dict，双向边的连接情况edge_index_dict用于信息传递，通过encoder得到更新之后的特征z_dict
        # 将z_dict和(drug, with, protein)的正负边输入decoder，得到关于边的连接情况
        z_dict = self.encoder(x_dict, edge_index_dict)
        pre = self.decoder(z_dict, edge_label_index, p_e_d)
        return pre, z_dict


def test(model, data, p_e_d):
    model.eval()
    with torch.no_grad():
        out, emb = model(data.x_dict, data.edge_index_dict,
                         data['drug', p_e_d].edge_label_index, p_e_d)
        out = out.view(-1).sigmoid()
        auc = roc_auc_score(data['drug', p_e_d].edge_label.cpu().numpy(), out.cpu().numpy())
        model.train()
    return auc, emb


def train(data, train_data, val_data, test_data, p_e_d, device):
    model = Model(hidden_channels_1=300, hidden_channels_2=300, data=data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    min_epochs = 10
    best_val_auc = 0
    final_test_auc = 0
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()

        out, embeddings_train = model(train_data.x_dict, train_data.edge_index_dict,
                                      train_data['drug', p_e_d].edge_label_index, p_e_d)
        out = out.view(-1)
        loss = criterion(out, train_data['drug', p_e_d].edge_label)
        loss.backward()
        optimizer.step()
        # validation
        val_auc, embeddings_val = test(model, val_data, p_e_d)
        test_auc, embeddings_test = test(model, test_data, p_e_d)
        if epoch + 1 > min_epochs and val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            final_embeddings = embeddings_test

        print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} test_auc {:.4f} '
              .format(epoch + 1, loss.item(), val_auc, test_auc))

    return final_test_auc, final_embeddings


# 将边集划分为训练集和测试集，其中test_ratio是测试集所占比例
def split_data(data, p_e_d):
    data = data.to(device)
    T.ToUndirected()(data)
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1,
        is_undirected=True,
        edge_types=[('drug', 'with', p_e_d), (p_e_d, 'rev_with', 'drug')],
    )(data)
    return train_data, val_data, test_data


kg_file = '../data/primeKG'
p_e_d = 'effect'
drug2id_file = '../data_after_processing/Heterogeneous_graph/drug2id'
df = pd.read_csv(kg_file, sep=',', header=0, low_memory=False)
triples = df.values.tolist()

rel_dict, drug2id, protein2id, effect2id, disease2id,   = {}, {}, {}, {}, {}
drug_related_protein, drug_related_effect, drug_related_disease = {}, {}, {}
drug_protein, drug_effect, drug_disease = [], [], []

for triple in triples:
    drug_1, rel, drug_2, x_type, y_type = triple[4], triple[0], triple[8], triple[3], triple[7]
    # 得到所有的drug字典
    if x_type == 'drug':
        _get_id(drug2id, drug_1)
    if y_type == 'drug':
        _get_id(drug2id, drug_2)

    # 统计不同类型的网络
    if rel == 'drug_protein':
        # 得到DPI网络中所有的蛋白质
        if x_type == 'gene/protein':
            protein_id = _get_id(protein2id, drug_1)
        if y_type == 'gene/protein':
            protein_id = _get_id(protein2id, drug_2)
        if x_type == 'drug':
            drug_id = _get_id(drug_related_protein, drug_1)
        if y_type == 'drug':
            drug_id = _get_id(drug_related_protein, drug_2)
        drug_protein.append([drug_id, protein_id])

    if rel == 'drug_effect':
        if x_type == 'effect/phenotype':
            effect_id = _get_id(effect2id, drug_1)
        if y_type == 'effect/phenotype':
            effect_id = _get_id(effect2id, drug_2)
        if x_type == 'drug':
            drug_id = _get_id(drug_related_effect, drug_1)
        if y_type == 'drug':
            drug_id = _get_id(drug_related_effect, drug_2)
        drug_effect.append([drug_id, effect_id])

    if rel in ["contraindication", "indication", "off-label use"]:
        if x_type == 'drug':
            drug_id = _get_id(drug_related_disease, drug_1)
        if y_type == 'drug':
            drug_id = _get_id(drug_related_disease, drug_2)
        if x_type == 'disease':
            disease_id = _get_id(disease2id, drug_1)
        if y_type == 'disease':
            disease_id = _get_id(disease2id, drug_2)
        drug_disease.append([drug_id, disease_id])

drugs = ["{}{}{}\n".format(val, '\t', key) for key, val in sorted(drug2id.items(), key=lambda x: x[1])]
with open(drug2id_file, "w+") as f:
    f.writelines(drugs)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# all_drug_features = torch.from_numpy(np.identity(num_all_drug, int))
all_drug_features = torch.randn(len(drug2id), 300)
if p_e_d == 'protein':
    related_drug = drug_related_protein
    p_e_d_dict = protein2id
    edges = drug_protein
if p_e_d == 'effect':
    related_drug = drug_related_effect
    p_e_d_dict = effect2id
    edges = drug_effect
if p_e_d == 'disease':
    related_drug = drug_related_disease
    p_e_d_dict = disease2id
    edges = drug_disease
data = built_heterograph(all_drug_features, drug2id, related_drug, p_e_d_dict, p_e_d, edges)
train_data, val_data, test_data = split_data(data, p_e_d)
auc, embeddings = train(data, train_data, val_data, test_data, p_e_d, device)
print(auc)
drug_embeddings = embeddings['drug'].cpu().numpy()
print(drug_embeddings.shape)

# 后面代码的目的是补全所有的drug的embedding,并保存
approved_drug_file = '../data/approved_structure links.csv'
p_e_d_related_file = 'embedding/{}_drug_embeddings.npy'.format(p_e_d)
p_e_d_drug_emb = []
with open(approved_drug_file) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        name = row[1]
        if name in related_drug.keys():
            drugid = related_drug[name]
            emb = drug_embeddings[drugid].reshape((1, 300))
        elif name in drug2id.keys():
            drugid = drug2id[name]
            emb = all_drug_features[drugid].reshape((1, 300))
        else:
            emb = np.zeros((1, 300))
        p_e_d_drug_emb.append(emb)
p_e_d_drug_emb = np.concatenate(p_e_d_drug_emb, axis=0)
print(p_e_d_drug_emb.shape)
np.save(p_e_d_related_file, p_e_d_drug_emb)

