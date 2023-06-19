import torch
import random
import numpy as np
import os
import torch.nn as nn
from model import My_Model
from utils import parse_args
import csv
from DataLoader import DataLoader


def get_drug2id(drug_file):
    drug_dict = {}
    id = 0
    with open(drug_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            name = row[1]
            drug_dict[name] = id
            id+=1
    return drug_dict


args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()  # 1
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
approved_drug_file = '../data/approved_structure links.csv'
structure_embed = np.load('../embedding/st_embedding.npy')
drug_protein_embed = np.load('../embedding/protein_drug_embeddings.npy')
drug_effect_embed = np.load('../embedding/effect_drug_embeddings.npy')
drug_disease_embed = np.load('../embedding/disease_drug_embeddings.npy')
kg_embed = np.load('../embedding/kg_{}_drug_embeddings.npy'.format(args.KGE_model))

structure_pre_embed = torch.tensor(structure_embed).float().to(device)
drug_protein_pre_embed = torch.tensor(drug_protein_embed).float().to(device)
drug_effect_pre_embed = torch.tensor(drug_effect_embed).float().to(device)
drug_disease_pre_embed = torch.tensor(drug_disease_embed).float().to(device)
kg_pre_embed = torch.tensor(kg_embed).float().to(device)
pre_embedding = [structure_pre_embed, drug_protein_pre_embed, drug_effect_pre_embed, drug_disease_pre_embed, kg_pre_embed]

ddi_file = open('../data_after_processing/case_study/{}_test_ddi.csv'.format(args.case_drug), 'r')
DDI_id_pair = []
label = []
for line in ddi_file.readlines():
    row = line.strip().split('\t')
    DDI_left_id = int(row[0])
    DDI_right_id = int(row[1])
    DDI_id_pair.append([DDI_left_id, DDI_right_id])
DDI_id_pair = np.array(DDI_id_pair)
test_x = torch.tensor(DDI_id_pair)
model = torch.load('../{}_model.pth'.format(args.case_drug))
model.to(device)
model.eval()
out = model('predict', test_x, pre_embedding)
out = out.squeeze(-1)
prediction = out.tolist()
print(prediction)
sorted_list = sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)
sorted_indices = [idx for idx, _ in sorted_list]
top = sorted_indices[:20]
res = []
drug_dict = get_drug2id(approved_drug_file)
case_drug_id = drug_dict[args.case_drug]
for index in top:
    prediction_value = prediction[index]
    if index >= case_drug_id:
        drug_id = index+1
    else:
        drug_id = index
    res.append((drug_id, prediction_value))

for item in res:
    print(item)
# DS1 38 45 58 62 64 77 82 83 86 88 91 93 95 96 98
#     Tryptophan Icosapent Ramipril Amphetamine Nicotine Troglitazone Succinylcholine Sildenafil Reserpine Ticlopidine Midodrine Torasemide Eletriptan Bethanidine Oxyphenonium
# Bortezomib Ritonavir Carbamazepine
