import numpy as np
import os
import torch
from dgl.nn.pytorch.glob import AvgPooling      # DGL 0.6.0 或更高版本pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ dgl==0.4.3
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem
import csv
os.chdir(r'/')


def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id


def get_drug_smile(approved_drug_file):
    graphs = []
    drug_map = {}
    with open(approved_drug_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            smile = row[6]
            name = row[1]
            _get_id(drug_map, name)
            if smile != '':
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    g = mol_to_bigraph(mol, add_self_loop=True,
                                       node_featurizer=PretrainAtomFeaturizer(),
                                       edge_featurizer=PretrainBondFeaturizer(),
                                       canonical_atom_order=False)
                    graphs.append(g)
                else:
                    graphs.append(None)
            else:
                graphs.append(None)

    return drug_map, graphs


def get_embedding(graphs):
    model = load_pretrained(model_name='gin_supervised_masking')
    model.eval()
    readout = AvgPooling()
    mol_emb = []
    for graph in graphs:
        if graph:
            nfeats = [graph.ndata.pop('atomic_number'),
                      graph.ndata.pop('chirality_type')]
            efeats = [graph.edata.pop('bond_type'),
                      graph.edata.pop('bond_direction_type')]
            with torch.no_grad():
                node_repr = model(graph, nfeats, efeats)
            # print(node_repr.shape)    # 如果某个化合物有20个原子，那么node_repr就是20*300，再经过readout之后就是1*300,作为化合物的表示
            mol_emb.append(readout(graph, node_repr))
        else:
            mol_emb.append(torch.zeros((1, 300)))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    print(mol_emb.shape)
    return mol_emb


drug2id_file = '../data_after_processing/st/drug2id.tsv'
approved_drug_file = '../data/approved_structure links.csv'  # FDA药物
store_file = '../embedding/st_embedding.npy'
st_drug, graphs = get_drug_smile(approved_drug_file)
drugs = ["{}{}{}\n".format(val, '\t', key) for key, val in sorted(st_drug.items(), key=lambda x: x[1])]
with open(drug2id_file, "w+") as f:
    f.writelines(drugs)
# 利用预训练得到的模型得到每个分子图的向量表示
mol_emb = get_embedding(graphs)
np.save(store_file, mol_emb)
