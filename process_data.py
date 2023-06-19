import pandas as pd
import torch
import rdkit
from rdkit.Chem import AllChem, DataStructs
from collections import Counter
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
from utils import parse_args
import csv
import json
import os


def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id


# ------------------------------------------从知识图谱中提取Kg数据和ddi数据--------------------------------------------------
def load_data_from_kg(kg_file, entity2id_file, relation2id_file, triple_file, ddi_file):
    kg_without_ddi, ddi_train_data = [], []
    entity_map, rel_map = {}, {}
    delimiter = '\t'

    df = pd.read_csv(kg_file, sep=',', header=0, low_memory=False)
    triples = df.values.tolist()
    for triple in triples:   # 先遍历一次
        drug_1, rel, drug_2 = triple[4], triple[0], triple[8]
        if rel != 'drug_drug':
            drug_1_id = _get_id(entity_map, drug_1)
            drug_2_id = _get_id(entity_map, drug_2)
            rel_id = _get_id(rel_map, rel)
            l1 = "{}{}{}{}{}\n".format(drug_1_id, delimiter, rel_id, delimiter, drug_2_id)
            kg_without_ddi.append(l1)
        else:   # 如果关系是ddi而且都能有st_embedding，那么可以得到ddi的训练数据
            ddi_pair = "{}{}{}\n".format(drug_1, "\t", drug_2)
            ddi_train_data.append(ddi_pair)

    with open(ddi_file, "w+") as f:
        f.writelines(ddi_train_data)

    entities = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(entity_map.items(), key=lambda x: x[1])]
    n_entities = len(entities)
    with open(entity2id_file, "w+", encoding='utf-8') as f:
        f.writelines(entities)

    relations = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(rel_map.items(), key=lambda x: x[1])]
    with open(relation2id_file, "w+", encoding='utf-8') as f:
        f.writelines(relations)
    n_relations = len(relations)

    with open(triple_file, "w+") as f:
        f.writelines(kg_without_ddi)
    n_triple = len(kg_without_ddi)

    print('{}个实体{}种关系{}个三元组'.format(n_entities, n_relations, n_triple))
    return n_triple


def generate_train_test_valid(kg_file, train_file, valid_file, test_file, num_triples):
    df = pd.read_csv(kg_file, sep="\t", header=None)
    triples = df.values.tolist()
    seed = np.arange(num_triples)
    np.random.shuffle(seed)
    train_cnt = int(num_triples * 0.9)
    valid_cnt = int(num_triples * 0.05)
    train_set = seed[:train_cnt]
    train_set = train_set.tolist()
    valid_set = seed[train_cnt:train_cnt + valid_cnt].tolist()
    test_set = seed[train_cnt + valid_cnt:].tolist()

    with open(train_file, 'w+') as f:
        for idx in train_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

    with open(valid_file, 'w+') as f:
        for idx in valid_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

    with open(test_file, 'w+') as f:
        for idx in test_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))


# ------------------------------------------------得到ddi正负样本数据(二分类和多分类）----------------------------------------
def get_drug2id(drug_file, v4_or_v5):
    drug_dict = {}
    with open(drug_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if v4_or_v5:
                name = row[0]
            else:
                name = row[1]
            _get_id(drug_dict, name)
        return drug_dict


def generate_pos_neg_data(drug_dict, DDI_pos_file, DDI_pos_neg_file, v4_or_v5=False):
    left_id = []
    right_id = []
    drug_type = set()
    with open(DDI_pos_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        if v4_or_v5:
            next(reader)
        for row in reader:
            left_name, right_name = row[0], row[1]
            if left_name in drug_dict and right_name in drug_dict:
                l_id, r_id = drug_dict[left_name], drug_dict[right_name]
                left_id.append(l_id)
                right_id.append(r_id)
                drug_type.add(l_id)
                drug_type.add(r_id)

    edge_list = [left_id, right_id]
    edge_index = torch.tensor(edge_list)
    num_pos_samples = edge_index.size(1)
    pos_label_index = torch.ones(num_pos_samples)
    data = Data(num_nodes=len(drug_dict), edge_index=edge_index)

    num_neg_samples = num_pos_samples
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=num_neg_samples, method='dense')
    neg_label_index = torch.zeros(num_neg_samples)
    pos_neg_label = torch.cat((pos_label_index, neg_label_index)).reshape(2*num_neg_samples, 1)
    pos_neg_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    pos_neg = pos_neg_index.T
    pair_and_label = torch.cat([pos_neg, pos_neg_label], dim=1).to(torch.int).tolist()
    count = 0
    with open(DDI_pos_neg_file, 'w') as f:
        for item in pair_and_label:
            drug_pair = str(item).replace('[', '').replace(']', '').replace(',', '\t')
            f.write(drug_pair+'\n')
            count += 1
    print('{}中共涉及到{}个药物的{}条ddi数据,通过负样本生成得到{}条正负边'.format(DDI_pos_file, len(drug_type), len(right_id), count))


def generate_multi_pos_neg_data(drug_file, multi_DDI_pos_file, multi_DDI_pos_neg_file):
    drug_map = {}
    label_map = {}
    with open(drug_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            name = row[0]
            _get_id(drug_map, name)
    ddi_pair = []
    type11 = []
    with open(multi_DDI_pos_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            left_name, right_name, label = row[0], row[1], row[2]
            if left_name in drug_map.keys() and right_name in drug_map.keys() and label not in ['1', '26', '42', '35']:
                l_id, r_id = drug_map[left_name], drug_map[right_name]
                type11.append(label)
                label_id = _get_id(label_map, label)
                ddi_pair.append((l_id, r_id, label_id))
    counter = Counter(type11)   # 有四种关系的ddi——pairs小于10
    print(counter)
    # with open(multi_DDI_pos_neg_file, 'w', newline='') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     for ddi in ddi_pair:
    #         writer.writerow(ddi)
    #     f.close()


# ------------------------------------------------得到chemicalx需要的文件--------------------------------------------------
def get_drug_with_fingerprint(approved_drug_file, drug_with_fingerprint_file):
    drug2features = {}
    name_smile = [['name', 'smile']]
    id = 0
    with open(approved_drug_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            smile = row[6]
            name = row[0]
            if smile != '':
                molecule = rdkit.Chem.MolFromSmiles(smile)
                if molecule:
                    fingerprint = AllChem.GetHashedMorganFingerprint(molecule, 2, nBits=300)
                    features = np.zeros((0,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fingerprint, features)
                    features = features.tolist()
                    drug2features[id] = {'smiles': row[6], 'features': features}
                    id += 1
                    name_smile.append([name, smile])
    with open(drug_with_fingerprint_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for drug in name_smile:
            writer.writerow(drug)
    return drug2features


def write_in(contrast_test_dir, drug2features, ddi_file):
    drug_set_file = os.path.join(contrast_test_dir, 'drug_set.json')
    drug_contexts_file = os.path.join(contrast_test_dir, 'context_set.json')
    triples_file = os.path.join(contrast_test_dir, 'labeled_triples.csv')
    # 写features,总共有2559个drug
    with open(drug_set_file, "w") as f:
        json.dump(drug2features, f)

    # 写context
    drug_contexts = {}  # :Mapping[str, Sequence[float]]
    for key in drug2features.keys():
        drug_contexts[key] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with open(drug_contexts_file, "w") as f:
        json.dump(drug_contexts, f)

    # 写ddi_pairs
    all_input_pair = [["drug_1", "drug_2", "context", "label"]]
    with open(ddi_file) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            drug1 = int(row[0])
            drug2 = int(row[1])
            label = float(row[2])
            all_input_pair.append([drug1, drug2, "context_01", label])
    with open(triples_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for pair in all_input_pair:
            writer.writerow(pair)


# ------------------------------------------------得到case_study需要的文件-------------------------------------------------
def get_case_drug_test_ddi(case_drug, drug_dict, case_drug_ddi_file):
    case_drug_id = drug_dict[case_drug]
    nums_drug = len(drug_dict)
    with open(case_drug_ddi_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for drug_id in range(nums_drug):
            if drug_id != case_drug_id:
                writer.writerow((case_drug_id, drug_id))


def get_case_drug_train_ddi(case_drug_id, all_ddi_file, out_file):
    ddi_without_case_drug = []
    pos_neg_ddi = open(all_ddi_file, 'r')
    for line in pos_neg_ddi.readlines():
        row = line.strip().split('\t')
        DDI_left_id = int(row[0])
        DDI_right_id = int(row[1])
        if DDI_left_id != case_drug_id and DDI_right_id != case_drug_id:
            ddi_pair = "{}{}{}{}{}\n".format(row[0], "\t", row[1], "\t", row[2])
            ddi_without_case_drug.append(ddi_pair)
    with open(out_file, 'w') as f:
        f.writelines(ddi_without_case_drug)


# 原始文件
args = parse_args()
kg = 'primeKG'  # primeKG或者drkg
kg_file = 'data/{}'.format(kg)
approved_drug_file = 'data/approved_structure links.csv'  # FDA药物
multi_label_file = 'data/multi_label_data.csv'

# 和kg有关的文件
entity2id_file = 'data_after_processing/kg/{}/entities.tsv'.format(kg)
relation2id_file = 'data_after_processing/kg/{}/relations.tsv'.format(kg)
triple_file = 'data_after_processing/kg/{}/all_triple.tsv'.format(kg)
train_file = "data_after_processing/kg/{}/kg_train.tsv".format(kg)
valid_file = "data_after_processing/kg/{}/kg_valid.tsv".format(kg)
test_file = "data_after_processing/kg/{}/kg_test.tsv".format(kg)

# 用于预测器训练的ddi文件
ddi_file = 'data_after_processing/predictor/ddi.txt'
pos_neg_ddi_file = 'data_after_processing/predictor/DDI_pos_neg.txt'
pos_neg_multi_ddi_file = 'data_after_processing/predictor/multi_DDI_pos_neg.tsv'

ddi_v4_file = 'data/ddi_v4.txt'
v4_pos_neg_ddi_file = 'data_after_processing/predictor/v4_DDI_pos_neg.txt'
ddi_v5_file = 'data/ddi_v5.txt'
v5_pos_neg_ddi_file = 'data_after_processing/predictor/v5_DDI_pos_neg.txt'


# # 首先为了避免标签泄露，需要删除kg数据中药物对之间的直连数据,st_drug_list1里的实体可以得到kg_embedding，删去了2672628个三元组,1269个实体
# n_triple = load_data_from_kg(kg_file, entity2id_file, relation2id_file, triple_file, ddi_file)

# # 得到dgl-ke所需要的三个文件
# generate_train_test_valid(triple_file, train_file, valid_file, test_file, n_triple)

# # 得到二分类需要的正负集
if args.ddi_dataset in ['DS2', 'DS3']:
    v4_or_v5 = True
else:
    v4_or_v5 =False
drug_dict = get_drug2id(approved_drug_file, v4_or_v5)

# generate_pos_neg_data(drug_dict, ddi_file, pos_neg_ddi_file, v4_or_v5)
# generate_pos_neg_data(drug_dict, ddi_v5_file, v5_pos_neg_ddi_file, v4_or_v5)
# generate_pos_neg_data(drug_dict, ddi_v4_file, v4_pos_neg_ddi_file, v4_or_v5)

# 得到多分类需要的正负集
# generate_multi_pos_neg_data(approved_drug_file, multi_label_file, pos_neg_multi_ddi_file)

# 得到对比试验的正负集
# contrast_test_dir = 'data_after_processing/contrast_test'
# drug_with_fingerprint_file = os.path.join(contrast_test_dir, 'drug_with_fingerprint.csv')
# compare_test_pos_neg_ddi_file = os.path.join(contrast_test_dir, 'compare_test_pos_neg_ddi.csv')
# drug2features = get_drug_with_fingerprint(approved_drug_file, drug_with_fingerprint_file)
# generate_pos_neg_data(drug_with_fingerprint_file, ddi_v5_file, compare_test_pos_neg_ddi_file, v4_or_v5=True)
# write_in(contrast_test_dir, drug2features, compare_test_pos_neg_ddi_file)

# 得到case_study所需要的文档
case_drug = args.case_drug       # ['Cannabidiol', 'Cisplatin', 'Dexamethasone']
case_drug_id = drug_dict[case_drug]
case_drug_train_file = 'data_after_processing/case_study/{}_train_ddi.txt'.format(case_drug)
case_drug_text_file = 'data_after_processing/case_study/{}_test_ddi.csv'.format(case_drug)

get_case_drug_train_ddi(case_drug_id, pos_neg_ddi_file, case_drug_train_file)
get_case_drug_test_ddi(case_drug, drug_dict, case_drug_text_file)

