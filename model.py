import torch.nn as nn
import torch
import torch.nn.functional as F


# 不同的情况下预测器的输入维度不一样
def get_input_dim(source, dim, fusion_type):
    if source in ['0', '1', '2', '3', '4']:
        drug_dim = dim
    if source in ['04']:
        if fusion_type == 'concat':
            drug_dim = dim * 2
        else:
            drug_dim = dim
    else:
        if fusion_type == 'concat':
            drug_dim = dim * 5
        else:
            drug_dim = dim
    pair_dim = drug_dim * 2
    return pair_dim


# main model
class My_Model(nn.Module):

    def __init__(self, args):
        super(My_Model, self).__init__()
        self.args = args

        self.self_attention_layer = torch.nn.MultiheadAttention(embed_dim=self.args.feature_dim, num_heads=1, batch_first=True)
        self.predictor_input_dim = get_input_dim(self.args.feature_source, self.args.feature_dim, self.args.fusion_type)

        # predictor使用的三层神经网络
        self.layer1 = nn.Sequential(nn.Linear(self.predictor_input_dim, self.args.n_hidden_1),
                                    nn.BatchNorm1d(self.args.n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(self.args.n_hidden_1, self.args.n_hidden_2),
                                    nn.BatchNorm1d(self.args.n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(self.args.n_hidden_2, self.args.out_dim))  # 输出为1维，后面会进行sigmoid操作

    def get_fusion_data(self, pre_embedding):
        if self.args.feature_source in ['0', '1', '2', '3', '4']:
            drug_embedding = pre_embedding[int(self.args.feature_source)]
        elif self.args.feature_source in ['04']:
            source_list = [int(source) for source in self.args.feature_source]
            embedding1 = pre_embedding[source_list[0]]
            embedding2 = pre_embedding[source_list[1]]
            embeddings = [embedding1, embedding2]
            drug_embedding = torch.cat(embeddings, dim=1)
        else:
            if self.args.fusion_type == 'concat':
                drug_embedding = torch.cat(pre_embedding, dim=1)
            elif self.args.fusion_type == 'mean':
                drug_embedding = (pre_embedding[0] + pre_embedding[1] + pre_embedding[2] +
                                  pre_embedding[3] + pre_embedding[4]) / 5
            elif self.args.fusion_type == 'element-wise':
                drug_embedding = (pre_embedding[0] * pre_embedding[1] * pre_embedding[2] *
                                  pre_embedding[3] * pre_embedding[4])
            elif self.args.fusion_type == 'attention':
                inputs = torch.stack(pre_embedding, dim=1)
                attn_output, _ = self.self_attention_layer(inputs, inputs, inputs)
                drug_embedding = attn_output.mean(dim=1)
        return drug_embedding

    def train_DDI_data(self, mode, train_data, pre_embedding):  # 训练DDI数据
        drug_embedding = self.get_fusion_data(pre_embedding)
        drug1 = train_data[:, 0].type(torch.long)
        drug2 = train_data[:, 1].type(torch.long)
        drug1_embed = drug_embedding[drug1]
        drug2_embed = drug_embedding[drug2]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def test_DDI_data(self, mode, test_data, pre_embedding):  # 训练DDI数据
        drug_embedding = self.get_fusion_data(pre_embedding)
        drug1 = test_data[:, 0].type(torch.long)
        drug2 = test_data[:, 1].type(torch.long)
        drug1_embed = drug_embedding[drug1]
        drug2_embed = drug_embedding[drug2]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.args.multi_class:
            res = F.softmax(x, dim=1)
        else:
            res = torch.sigmoid(x)
        return res

    def forward(self, mode, *input):
        if mode == 'calc_ddi_loss':
            return self.train_DDI_data(mode, *input)
        if mode == 'predict':
            return self.test_DDI_data(mode, *input)
