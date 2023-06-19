import os
import gc
import time
import csv
import json
import torch as th
import numpy as np
import scipy as sp
import logging
import argparse
import dgl.backend as F
from dgl.base import NID, EID
import dgl
import torch.nn as nn
import torch.nn.init as INIT
import torch.nn.functional as functional
import torch.multiprocessing as mp
from torch.multiprocessing import Queue


DEFAULT_INFER_BATCHSIZE = 2048
EMB_INIT_EPS = 2.0
logsigmoid = functional.logsigmoid  # 映射到0.1之间
none = lambda x: x
norm = lambda x, p: x.norm(p=p) ** p
get_scalar = lambda x: x.detach().item()    # 返回一个标量
reshape = lambda arr, x, y: arr.view(x, y)
cuda = lambda arr, gpu: arr.cuda(gpu)


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--model_name', default='RotatE',
                          choices=['TransE_l1', 'TransE_l2', 'DistMult', 'ComplEx', 'RotatE', 'SimplE'],
                          help='The models provided by DGL-KE.')
        self.add_argument('--dataset', type=str, default='primeKG',
                          help='The name of the builtin knowledge graph. Currently, the builtin knowledge ' \
                               'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. ' \
                               'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
        self.add_argument('--data_path', type=str, default='data_after_processing/kg/primeKG/',
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--data_files', type=str,
                          default=['entities.tsv', 'relations.tsv', 'kg_train.tsv', 'kg_valid.tsv', 'kg_test.tsv'],
                          nargs='+',
                          help='A list of data file names. This is used if users want to train KGE' \
                               'on their own datasets. If the format is raw_udd_{htr},' \
                               'users need to provide train_file [valid_file] [test_file].' \
                               'If the format is udd_{htr}, users need to provide' \
                               'entity_file relation_file train_file [valid_file] [test_file].' \
                               'In both cases, valid_file and test_file are optional.')
        self.add_argument('--format', type=str, default='udd_hrt',
                          help='The format of the dataset. For builtin knowledge graphs,' \
                               'the foramt should be built_in. For users own knowledge graphs,' \
                               'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--batch_size', type=int, default=2048,
                          help='The batch size for training.')
        self.add_argument('--batch_size_eval', type=int, default=16,
                          help='The batch size used for validation and case_study.')
        self.add_argument('--neg_sample_size', type=int, default=128,
                          help='The number of negative samples we use for each positive sample in the training.')
        self.add_argument('--hidden_dim', type=int, default=150,
                          help='The embedding size of relation and entity')
        self.add_argument('--lr', type=float, default=0.1,
                          help='The learning rate. DGL-KE uses Adagrad to optimize the model parameters.')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='The margin value in the score function. It is used by TransX and RotatE.')
        self.add_argument('--max_step', type=int, default=100000,
                          help='The maximal number of steps to train the model.' \
                               'A step trains the model with a batch of data.')
        self.add_argument('-log', '--log_interval', type=int, default=1000,
                          help='Print runtime of different components every x steps.')
        self.add_argument('-rc', '--regularization_coef', type=float, default=1.00E-07,
                          help='The coefficient for regularization.')
        self.add_argument('-adv', '--neg_adversarial_sampling', action='store_true', default=True,
                          help='Indicate whether to use negative adversarial sampling.' \
                               'It will weight negative samples with higher scores more.')
        self.add_argument('--case_study', action='store_true', default=True,
                          help='Evaluate the model on the case_study set after the model is trained.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to train the model in parallel.' \
                               'In multi-GPU training, the number of processes by default is set to match the number of GPUs.' \
                               'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to train the model in each process.' \
                               'This argument is used for multiprocessing training.')
        self.add_argument('--async_update', action='store_true', default=False,     # 只有一个gpu的时候就没必要开异步更新
                          help='Allow asynchronous update on node embedding for multi-GPU training.' \
                               'This overlaps CPU and GPU computation to speed up.')
        self.add_argument('--gpu', type=int, default=[0], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--neg_sample_size_eval', type=int, default=10000,
                          help='The number of negative samples we use to evaluate a positive sample.')
        self.add_argument('-de', '--double_ent', action='store_true', default=False,
                          help='Double entitiy dim for complex number It is used by RotatE.')
        self.add_argument('-dr', '--double_rel', action='store_true', default=False,
                          help='Double relation dim for complex number.')

        self.add_argument('--delimiter', type=str, default='\t',
                          help='Delimiter used in data files. Note all files should use the same delimiter.')
        self.add_argument('--save_path', type=str, default='kg_models',
                          help='the path of the directory where models and logs are saved.')
        self.add_argument('--no_save_emb', action='store_true',
                          help='Disable saving the embeddings under save_path.')
        self.add_argument('--neg_deg_sample', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the training.' \
                               'When this option is turned on, the number of negative samples per positive edge' \
                               'will be doubled. Half of the negative samples are generated uniformly while' \
                               'the other half are generated proportional to vertex degree.')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the evaluation.')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='Randomly sample some percentage of edges for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='Disable filter positive edges from randomly constructed negative edges for evaluation')
        self.add_argument('--eval_interval', type=int, default=10000,
                          help='Print evaluation results on the validation dataset every x steps' \
                               'if validation is turned on')
        self.add_argument('--force_sync_interval', type=int, default=-1,
                          help='We force a synchronization between processes every x steps for' \
                               'multiprocessing training. This potentially stablizes the training process'
                               'to get a better performance. For multiprocessing training, it is set to 1000 by default.')
        self.add_argument('-a', '--adversarial_temperature', default=1.0, type=float,
                          help='The temperature used for negative adversarial sampling.')
        self.add_argument('-rn', '--regularization_norm', type=int, default=3,
                          help='norm used in regularization.')
        self.add_argument('--mix_cpu_gpu', action='store_true', default=False,
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.' \
                               'The embeddings are stored in CPU memory and the training is performed in GPUs.' \
                               'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--rel_part', action='store_true',
                          help='Enable relation partitioning for multi-GPU training.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.' \
                               'The positive score will be adjusted ' \
                               'as pos_score = pos_score * edge_importance')


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_'.format(args.model_name, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])  # 取指定目录下以指定字符串开头的文件数量，并将其赋值给变量 n
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


def get_device(args):
    return th.device('cpu') if args.gpu[0] < 0 else th.device('cuda:' + str(args.gpu[0]))


def save_model(args, model, emap_file=None, rmap_file=None):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print('Save model to {}'.format(args.save_path))
    model.save_emb(args.save_path, args.dataset)

    # We need to save the model configurations as well.
    conf_file = os.path.join(args.save_path, 'config.json')
    dict = {}
    config = args
    dict.update(vars(config))
    dict.update({'emp_file': emap_file,
                 'rmap_file': rmap_file})
    with open(conf_file, 'w') as outfile:
        json.dump(dict, outfile, indent=4)
    return args.save_path


# ---------------------------------------------加载数据-------------------------------------------------------------------
class KGDataset:
    '''Load a knowledge graph

    The folder with a knowledge graph has five files:
    * entities stores the mapping between entity Id and entity name.
    * relations stores the mapping between relation Id and relation name.
    * train stores the triples in the training set.
    * valid stores the triples in the validation set.
    * case_study stores the triples in the case_study set.

    The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.

    The triples are stored as 'head_name\trelation_name\ttail_name'.
    '''

    def __init__(self, entity_path, relation_path, train_path,
                 valid_path=None, test_path=None, format=[0, 1, 2],
                 delimiter='\t', skip_first_line=False):
        self.delimiter = delimiter
        self.entity2id, self.n_entities = self.read_entity(entity_path)
        self.relation2id, self.n_relations = self.read_relation(relation_path)
        self.train = self.read_triple(train_path, "train", skip_first_line, format)
        if valid_path is not None:
            self.valid = self.read_triple(valid_path, "valid", skip_first_line, format)
        else:
            self.valid = None
        if test_path is not None:
            self.test = self.read_triple(test_path, "case_study", skip_first_line, format)
        else:
            self.test = None

    def read_entity(self, entity_path):
        with open(entity_path) as f:
            entity2id = {}
            for line in f:
                eid, entity = line.strip().split(self.delimiter)
                entity2id[entity] = int(eid)

        return entity2id, len(entity2id)

    def read_relation(self, relation_path):
        with open(relation_path) as f:
            relation2id = {}
            for line in f:
                rid, relation = line.strip().split(self.delimiter)
                relation2id[relation] = int(rid)

        return relation2id, len(relation2id)

    def read_triple(self, path, mode, skip_first_line=False, format=[0, 1, 2]):
        # mode: train/valid/case_study
        if path is None:
            return None

        print('Reading {} triples....'.format(mode))
        heads = []
        tails = []
        rels = []
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split(self.delimiter)
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                heads.append(self.entity2id[h])
                rels.append(self.relation2id[r])
                tails.append(self.entity2id[t])

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))

        return heads, rels, tails


class KGDatasetUDD(KGDataset):  # KGDataset是KGDatasetUDD的父类，可以调用父类中的相关函数
    '''Load a knowledge graph user defined dataset

    The user defined dataset has five files:
    * entities stores the mapping between entity name and entity Id.
    * relations stores the mapping between relation name relation Id.
    * train stores the triples in the training set. In format [src_id, rel_id, dst_id]
    * valid stores the triples in the validation set. In format [src_id, rel_id, dst_id]
    * case_study stores the triples in the case_study set. In format [src_id, rel_id, dst_id]

    The mapping between entity (relation) name and entity (relation) Id is stored as 'name\tid'.
    The triples are stored as 'head_nid\trelation_id\ttail_nid'. Users can also use other delimiters
    other than \t.
    '''

    def __init__(self, path, name, delimiter, files, format, has_edge_importance=False):
        self.name = name
        format = [0, 1, 2]
        self.has_edge_importance = has_edge_importance
        super(KGDatasetUDD, self).__init__(os.path.join(path, files[0]),  # super代表调用父类的用法
                                           os.path.join(path, files[1]),
                                           os.path.join(path, files[2]),
                                           os.path.join(path, files[3]),
                                           os.path.join(path, files[4]),
                                           format=format,
                                           delimiter=delimiter)
        self.emap_file = files[0]
        self.rmap_file = files[1]

    def read_entity(self, entity_path):
        n_entities = 0
        with open(entity_path, encoding='utf-8') as f_ent:
            for line in f_ent:
                n_entities += 1
        return None, n_entities

    def read_relation(self, relation_path):
        n_relations = 0
        with open(relation_path, encoding='utf-8') as f_rel:
            for line in f_rel:
                n_relations += 1
        return None, n_relations

    def read_triple(self, path, mode, skip_first_line=False, format=[0, 1, 2]):
        heads = []
        tails = []
        rels = []
        e_impts = []
        print('Reading {} triples....'.format(mode))
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split(self.delimiter)
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                try:
                    heads.append(int(h))
                    tails.append(int(t))
                    rels.append(int(r))
                    if self.has_edge_importance:
                        e_impts.append(float(triple[3]))
                except ValueError:
                    print("For User Defined Dataset, both node ids and relation ids in the " \
                          "triplets should be int other than {}\t{}\t{}".format(h, r, t))
                    raise
        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))
        assert np.max(heads) < self.n_entities, \
            'Head node ID should not exceeds the number of entities {}'.format(self.n_entities)
        assert np.max(tails) < self.n_entities, \
            'Tail node ID should not exceeds the number of entities {}'.format(self.n_entities)
        assert np.max(rels) < self.n_relations, \
            'Relation ID should not exceeds the number of relations {}'.format(self.n_relations)

        assert np.min(heads) >= 0, 'Head node ID should >= 0'
        assert np.min(tails) >= 0, 'Tail node ID should >= 0'
        assert np.min(rels) >= 0, 'Relation ID should >= 0'

        if self.has_edge_importance:
            e_impts = np.array(e_impts, dtype=np.float)
            assert np.min(e_impts) > 0., 'Edge importance score should > 0'
            return (heads, rels, tails, e_impts)
        return (heads, rels, tails)

    @property
    def emap_fname(self):
        return self.emap_file

    @property
    def rmap_fname(self):
        return self.rmap_file


def get_dataset(data_path, data_name, format_str, delimiter='\t', files=None, has_edge_importance=False):
    format = format_str[4:]
    dataset = KGDatasetUDD(data_path, data_name, delimiter, files, format, has_edge_importance)
    return dataset


# ---------------------------------------------生成图并进行负采样----------------------------------------------------------
def ConstructGraph(dataset, args):
    """Construct Graph for training

    Parameters
    ----------
    dataset :
        the dataset
    args :
        Global configs.
    """
    src = [dataset.train[0]]
    etype_id = [dataset.train[1]]
    dst = [dataset.train[2]]
    num_train = len(dataset.train[0])

    if hasattr(dataset, 'valid') and dataset.valid is not None:
        src.append(dataset.valid[0])
        etype_id.append(dataset.valid[1])
        dst.append(dataset.valid[2])
        num_valid = len(dataset.valid[0])
    else:
        num_valid = 0

    if hasattr(dataset, 'case_study') and dataset.test is not None:
        src.append(dataset.test[0])
        etype_id.append(dataset.test[1])
        dst.append(dataset.test[2])
        num_test = len(dataset.test[0])
    else:
        num_test = 0

    src = np.concatenate(src)
    etype_id = np.concatenate(etype_id)
    dst = np.concatenate(dst)
    n_entities = dataset.n_entities

    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
    g.edata['tid'] = F.tensor(etype_id, F.int64)
    return g


# 用于训练数据
class TrainDataset(object):
    """Dataset for training

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    ranks:
        Number of partitions.
    """

    def __init__(self, g, dataset, args, ranks=64, has_importance=False):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)
        self.edge_parts = [np.arange(num_train)]
        self.rel_parts = [np.arange(dataset.n_relations)]
        self.cross_part = False

        self.g = g

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=32,
                       shuffle=True, exclude_positive=False, rank=0):
        """Create sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        exclude_positive : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        assert batch_size % neg_sample_size == 0, 'batch_size should be divisible by B'
        # EdgeSampler返回的是一个tuple，包含两个元素，分别是采样得到的源节点的ID数组和目标节点的ID数组
        return EdgeSampler(self.g,
                           seed_edges=F.tensor(self.edge_parts[rank]).long(),
                           batch_size=batch_size,
                           neg_sample_size=int(neg_sample_size / neg_chunk_size),  # 1 意思就是每一个1：1进行采样
                           chunk_size=neg_chunk_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)


# 用于测试或者验证集
class EvalSampler(object):
    """Sampler for validation and testing

    Parameters
    ----------
    g : DGLGraph
        Graph containing KG graph
    edges : tensor
        Seed edges
    batch_size : int
        Batch size of each mini batch.
    neg_sample_size : int
        How many negative edges sampled for each node.
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    mode : str
        Sampling mode.
    number_workers: int
        Number of workers used in parallel for this sampler
    filter_false_neg : bool
        If True, exlucde true positive edges in sampled negative edges
        If False, return all sampled negative edges even there are positive edges
        Default: True
    """

    def __init__(self, g, edges, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers=32,
                 filter_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   chunk_size=neg_chunk_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['tid'],
                                   return_false_neg=filter_false_neg)
        self.sampler_iter = iter(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g
        self.filter_false_neg = filter_false_neg
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch

        Returns
        -------
        DGLGraph
            Sampled positive graph
        ChunkNegEdgeSubgraph
            Negative graph wrapper
        """
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g,
                                        self.neg_chunk_size,
                                        self.neg_sample_size,
                                        'chunk' in self.mode,
                                        self.neg_head,
                                        self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

    def reset(self):
        """Reset the sampler
        """
        self.sampler_iter = iter(self.sampler)
        return self


class EvalDataset(object):
    """Dataset for validation or testing

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """

    def __init__(self, g, dataset, args):
        self.num_train = len(dataset.train[0])
        self.num_valid = len(dataset.valid[0])
        self.num_test = len(dataset.test[0])
        self.g = g
        self.valid = np.arange(self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))
        self.test = np.arange(self.num_train + self.num_valid, self.g.number_of_edges())
        print('|case_study|:', len(self.test))

    def get_edges(self, eval_type):
        """ Get all edges in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'case_study' for testing

        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'case_study':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def get_dicts(self, eval_type):
        """ Get all edges dict in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'case_study' for testing

        Returns
        -------
        dict
            all edges
        """
        if eval_type == 'valid':
            return self.valid_dict
        elif eval_type == 'case_study':
            return self.test_dict
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=32, rank=0, ranks=1):
        """Create sampler for validation or testing

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'case_study' for testing
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        filter_false_neg : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks  # 0
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])  # 276221
        edges = edges[beg: end]  # 取了所有的test边
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)


# 负采样
class ChunkNegEdgeSubgraph(dgl.DGLGraph):
    """Wrapper for negative graph

        Parameters
        ----------
        neg_g : DGLGraph
            Graph holding negative edges.
        num_chunks : int
            Number of chunks in sampled graph.
        chunk_size : int
            Info of chunk_size.
        neg_sample_size : int
            Info of neg_sample_size.
        neg_head : bool
            If True, negative_mode is 'head'
            If False, negative_mode is 'tail'
    """

    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
                                                   readonly=True,
                                                   parent=subg._parent)
        self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
        self.edata[EID] = subg.sgi.induced_edges.tousertensor()
        self.subg = subg
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head

    @property
    def head_nid(self):
        return self.subg.head_nid

    @property
    def tail_nid(self):
        return self.subg.tail_nid


def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
                        neg_head, num_nodes):
    """KG models need to know the number of chunks, the chunk size and negative sample size
    of a negative subgraph to perform the computation more efficiently.
    This function tries to infer all of these information of the negative subgraph
    and create a wrapper class that contains all of the information.

    Parameters
    ----------
    pos_g : DGLGraph
        Graph holding positive edges.
    neg_g : DGLGraph
        Graph holding negative edges.
    chunk_size : int
        Chunk size of negative subgrap.
    neg_sample_size : int
        Negative sample size of negative subgrap.
    is_chunked : bool
        If True, the sampled batch is chunked.
    neg_head : bool
        If True, negative_mode is 'head'
        If False, negative_mode is 'tail'
    num_nodes: int
        Total number of nodes in the whole graph.

    Returns
    -------
    ChunkNegEdgeSubgraph
        Negative graph wrapper
    """
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
            or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        # This is probably for evaluation.
        if pos_g.number_of_edges() < chunk_size \
                and neg_g.number_of_edges() % neg_sample_size == 0:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        # This is probably the last batch in the training. Let's ignore it.
        elif pos_g.number_of_edges() % chunk_size > 0:
            return None
        else:
            num_chunks = int(pos_g.number_of_edges() / chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)


class NewBidirectionalOneShotIterator:
    """Grouped sampler iterator

    Parameters
    ----------
    dataloader_head : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in head mode
    dataloader_tail : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in tail mode
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    neg_sample_size : int
        How many negative edges sampled for each node.
    is_chunked : bool
        If True, the sampled batch is chunked.
    num_nodes : int
        Total number of nodes in the whole graph.
    """

    def __init__(self, dataloader_head, dataloader_tail, neg_chunk_size, neg_sample_size,
                 is_chunked, num_nodes, has_edge_importance=False):
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    True, num_nodes, has_edge_importance)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    False, num_nodes, has_edge_importance)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_g, neg_g = next(self.iterator_head)
        else:
            pos_g, neg_g = next(self.iterator_tail)
        return pos_g, neg_g

    @staticmethod
    def one_shot_iterator(dataloader, neg_chunk_size, neg_sample_size, is_chunked,
                          neg_head, num_nodes, has_edge_importance=False):
        while True:
            for pos_g, neg_g in dataloader:
                neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
                                            is_chunked, neg_head, num_nodes)
                if neg_g is None:
                    continue

                pos_g.ndata['id'] = pos_g.parent_nid
                neg_g.ndata['id'] = neg_g.parent_nid
                pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
                if has_edge_importance:
                    pos_g.edata['impts'] = pos_g._parent.edata['impts'][pos_g.parent_eid]
                yield pos_g, neg_g


# ----------------------------------------------------model-------------------------------------------------------------
def async_update(args, emb, queue):
    """Asynchronous embedding update for entity embeddings.
    How it works:
        1. trainer process push entity embedding update requests into the queue.
        2. async_update process pull requests from the queue, calculate
           the gradient state and gradient and write it into entity embeddings.

    Parameters
    ----------
    args :
        Global confis.
    emb : ExternalEmbedding
        The entity embeddings.
    queue:
        The request queue.
    """
    th.set_num_threads(args.num_thread)
    while True:
        (grad_indices, grad_values, gpu_id) = queue.get()
        clr = emb.args.lr
        if grad_indices is None:
            return
        with th.no_grad():
            grad_sum = (grad_values * grad_values).mean(1)
            device = emb.state_sum.device
            if device != grad_indices.device:
                grad_indices = grad_indices.to(device)
            if device != grad_sum.device:
                grad_sum = grad_sum.to(device)

            emb.state_sum.index_add_(0, grad_indices, grad_sum)
            std = emb.state_sum[grad_indices]  # _sparse_mask
            if gpu_id >= 0:
                std = std.cuda(gpu_id)
            std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
            tmp = (-clr * grad_values / std_values)
            if tmp.device != device:
                tmp = tmp.to(device)
            emb.emb.index_add_(0, grad_indices, tmp)


class ExternalEmbedding:
    """Sparse Embedding for Knowledge Graph
    It is used to store both entity embeddings and relation embeddings.

    Parameters
    ----------
    args :
        Global configs.
    num : int
        Number of embeddings.
    dim : int
        Embedding dimention size.
    device : th.device
        Device to store the embedding.
    """

    def __init__(self, args, num, dim, device):
        self.gpu = args.gpu
        self.args = args
        self.num = num
        self.trace = []

        self.emb = th.empty(num, dim, dtype=th.float32, device=device)      # 创建一个空的tensor
        self.state_sum = self.emb.new().resize_(self.emb.size(0)).zero_()
        self.state_step = 0
        self.has_cross_rel = False
        # queue used by asynchronous update
        self.async_q = None
        # asynchronous update process
        self.async_p = None

    # 初始化嵌入
    def init(self, emb_init):
        """Initializing the embeddings.

        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
        INIT.uniform_(self.emb, -emb_init, emb_init)        # 随机初始化
        INIT.zeros_(self.state_sum)                         # 0值初始化

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process tensor access
        """
        self.emb.share_memory_()
        self.state_sum.share_memory_()

    def __call__(self, idx, gpu_id=-1, trace=True):
        """ Return sliced tensor.

        Parameters
        ----------
        idx : th.tensor
            Slicing index
        gpu_id : int
            Which gpu to put sliced data in.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: True
        """
        if self.has_cross_rel:
            cpu_idx = idx.cpu()
            cpu_mask = self.cpu_bitmap[cpu_idx]
            cpu_idx = cpu_idx[cpu_mask]
            cpu_idx = th.unique(cpu_idx)
            if cpu_idx.shape[0] != 0:
                cpu_emb = self.global_emb.emb[cpu_idx]
                self.emb[cpu_idx] = cpu_emb.cuda(gpu_id)
        s = self.emb[idx]
        if gpu_id >= 0:
            s = s.cuda(gpu_id)
        # During the training, we need to trace the computation.
        # In this case, we need to record the computation path and compute the gradients.
        if trace:
            data = s.clone().detach().requires_grad_(True)
            self.trace.append((idx, data))      # 返回实体id和它对应的embedding
        else:
            data = s
        return data

    # 更新嵌入
    def update(self, gpu_id=-1):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        self.state_step += 1
        for idx, data in self.trace:
            grad = data.grad.data

            clr = self.args.lr
            # clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])

            # the update is non-linear so indices must be unique
            grad_indices = idx
            grad_values = grad
            if self.async_q is not None:
                grad_indices.share_memory_()
                grad_values.share_memory_()
                self.async_q.put((grad_indices, grad_values, gpu_id))
            else:
                grad_sum = (grad_values * grad_values).mean(1)
                device = self.state_sum.device
                if device != grad_indices.device:
                    grad_indices = grad_indices.to(device)
                if device != grad_sum.device:
                    grad_sum = grad_sum.to(device)

                if self.has_cross_rel:
                    cpu_mask = self.cpu_bitmap[grad_indices]
                    cpu_idx = grad_indices[cpu_mask]
                    if cpu_idx.shape[0] > 0:
                        cpu_grad = grad_values[cpu_mask]
                        cpu_sum = grad_sum[cpu_mask].cpu()
                        cpu_idx = cpu_idx.cpu()
                        self.global_emb.state_sum.index_add_(0, cpu_idx, cpu_sum)
                        std = self.global_emb.state_sum[cpu_idx]
                        if gpu_id >= 0:
                            std = std.cuda(gpu_id)
                        std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                        tmp = (-clr * cpu_grad / std_values)
                        tmp = tmp.cpu()
                        self.global_emb.emb.index_add_(0, cpu_idx, tmp)
                self.state_sum.index_add_(0, grad_indices, grad_sum)
                std = self.state_sum[grad_indices]  # _sparse_mask
                if gpu_id >= 0:
                    std = std.cuda(gpu_id)
                std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                tmp = (-clr * grad_values / std_values)
                if tmp.device != device:
                    tmp = tmp.to(device)
                # TODO(zhengda) the overhead is here.
                self.emb.index_add_(0, grad_indices, tmp)
        self.trace = []

    # 在正则化的时候会用到
    def curr_emb(self):
        """Return embeddings in trace.
        """
        data = [data for _, data in self.trace]
        return th.cat(data, 0)

    # 存取embedding
    def save(self, path, name):
        """Save embeddings.

        Parameters
        ----------
        path : str
            Directory to save the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name + '.npy')
        np.save(file_name, self.emb.cpu().detach().numpy())

    def load(self, path, name):
        """Load embeddings.

        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name + '.npy')
        self.emb = th.Tensor(np.load(file_name))

    # 后面的函数都用不到，原因是我只用了一个gpu
    def setup_cross_rels(self, cross_rels, global_emb):
        cpu_bitmap = th.zeros((self.num,), dtype=th.bool)
        for i, rel in enumerate(cross_rels):
            cpu_bitmap[rel] = 1
        self.cpu_bitmap = cpu_bitmap
        self.has_cross_rel = True
        self.global_emb = global_emb

    def get_noncross_idx(self, idx):
        cpu_mask = self.cpu_bitmap[idx]
        gpu_mask = ~cpu_mask
        return idx[gpu_mask]

    def create_async_update(self):
        """Set up the async update subprocess.
        """
        self.async_q = Queue(1)
        self.async_p = mp.Process(target=async_update, args=(self.args, self, self.async_q))
        self.async_p.start()

    def finish_async_update(self):
        """Notify the async update subprocess to quit.
        """
        self.async_q.put((None, None, None))
        self.async_p.join()


#                             ----下面是不同kge-model的三元组评分指标----
def batched_l2_dist(a, b):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = th.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res


# transe-l1
def batched_l1_dist(a, b):
    res = th.cdist(a, b, p=1)
    return res


class TransEScore(nn.Module):
    """ TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """
    def __init__(self, gamma, dist_func='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        if dist_func == 'l1':
            self.neg_dist_func = batched_l1_dist
            self.dist_ord = 1
        else: # default use l2
            self.neg_dist_func = batched_l2_dist
            self.dist_ord = 2

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=self.dist_ord, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.expand_dims(axis=1)
        rel_emb = rel_emb.expand_dims(axis=0)
        score = (head_emb + rel_emb).expand_dims(axis=2) - tail_emb.expand_dims(axis=0).expand_dims(axis=0)

        return self.gamma - th.norm(score, p=self.dist_ord, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, chunk_size, hidden_dim)
                return gamma - self.neg_dist_func(tails, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads + relations
                heads = heads.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                return gamma - self.neg_dist_func(heads, tails)
            return fn


# DistMult对应的损失函数
class DistMultScore(nn.Module):
    """DistMult score function
    Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb * rel_emb).unsqueeze(2) * tail_emb.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, dim0=1, dim1=2)
                tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, dim0=1, dim1=2)
                tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.matmul(tmp, tails)
            return fn


# ComplEx对应的损失函数
class ComplExScore(nn.Module):
    """ComplEx score function
    Paper link: https://arxiv.org/abs/1606.06357
    """
    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = th.split(edges.src['emb'], split_size_or_sections=edges.src['emb'].shape[1]//2, dim=-1)
        real_tail, img_tail = th.split(edges.dst['emb'], split_size_or_sections=edges.dst['emb'].shape[1]//2, dim=-1)
        real_rel, img_rel = th.split(edges.data['emb'], split_size_or_sections=edges.data['emb'].shape[1]//2, dim=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        real_head, img_head = th.split(head_emb, split_size_or_sections=head_emb.shape[1]//2, dim=-1)
        real_tail, img_tail = th.split(tail_emb, split_size_or_sections=tail_emb.shape[1]//2, dim=-1)
        real_rel, img_rel = th.split(rel_emb, split_size_or_sections=rel_emb.shape[1]//2, dim=-1)

        score = (real_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0) \
                + (img_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
                + (real_head.unsqueeze(1) * img_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
                - (img_head.unsqueeze(1) * img_rel.unsqueeze(0)).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real, emb_img = th.split(tails, split_size_or_sections=tails.shape[1]//2, dim=-1)
                rel_real, rel_img = th.split(relations, split_size_or_sections=relations.shape[1]//2, dim=-1)
                real = emb_real * rel_real + emb_img * rel_img
                img = -emb_real * rel_img + emb_img * rel_real
                emb_complex = th.cat((real, img), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, dim0=1, dim1=2)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real, emb_img = th.split(heads, split_size_or_sections=heads.shape[1]//2, dim=-1)
                rel_real, rel_img = th.split(relations, split_size_or_sections=relations.shape[1]//2, dim=-1)
                real = emb_real * rel_real - emb_img * rel_img
                img = emb_real * rel_img + emb_img * rel_real
                emb_complex = th.cat((real, img), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)

                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, dim0=1, dim1=2)
                return th.bmm(tmp, tails)
            return fn


# RotatE对应的损失函数
class RotatEScore(nn.Module):
    """RotatE score function
    Paper link: https://arxiv.org/abs/1902.10197
    """
    def __init__(self, gamma, emb_init):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):
        re_head, im_head = th.chunk(edges.src['emb'], 2, dim=-1)
        re_tail, im_tail = th.chunk(edges.dst['emb'], 2, dim=-1)

        phase_rel = edges.data['emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return {'score': self.gamma - score.sum(-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        re_head, im_head = th.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = th.chunk(tail_emb, 2, dim=-1)

        phase_rel = rel_emb / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head.unsqueeze(1) * re_rel.unsqueeze(0) - im_head.unsqueeze(1) * im_rel.unsqueeze(0)
        im_score = re_head.unsqueeze(1) * im_rel.unsqueeze(0) + im_head.unsqueeze(1) * re_rel.unsqueeze(0)

        re_score = re_score.unsqueeze(2) - re_tail.unsqueeze(0).unsqueeze(0)
        im_score = im_score.unsqueeze(2) - im_tail.unsqueeze(0).unsqueeze(0)
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return self.gamma - score.sum(-1)

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))  # 得到g.edata['score']

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        emb_init = self.emb_init
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - heads
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)
                return gamma - score.sum(-1)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real

                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - tails
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)

                return gamma - score.sum(-1)

            return fn


# SimplE对应的损失函数
class SimplEScore(nn.Module):
    """SimplE score function
    Paper link: http://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    """
    def __init__(self):
        super(SimplEScore, self).__init__()

    def edge_func(self, edges):
        head_i, head_j = th.chunk(edges.src['emb'], 2, dim=-1)
        tail_i, tail_j = th.chunk(edges.dst['emb'], 2, dim=-1)
        rel, rel_inv = th.chunk(edges.data['emb'], 2, dim=-1)
        forward_score = head_i * rel * tail_j
        backward_score = tail_i * rel_inv * head_j
        # clamp as official implementation does to avoid NaN output
        # might because of gradient explode
        score = th.clamp(1 / 2 * (forward_score + backward_score).sum(-1), -20, 20)
        return {'score': score}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_i, head_j = th.chunk(head_emb.unsqueeze(1), 2, dim=-1)
        tail_i, tail_j = th.chunk(tail_emb.unsqueeze(0).unsqueeze(0), 2, dim=-1)
        rel, rel_inv = th.chunk(rel_emb.unsqueeze(0), 2, dim=-1)
        forward_tmp = (head_i * rel).unsqueeze(2) * tail_j
        backward_tmp = (head_j * rel_inv).unsqueeze(2) * tail_i
        score = (forward_tmp + backward_tmp) * 1 / 2
        return th.sum(score, dim=-1)

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tail_i = tails[..., :hidden_dim // 2]
                tail_j = tails[..., hidden_dim // 2:]
                rel = relations[..., : hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (rel * tail_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * tail_i).reshape(num_chunks, chunk_size, hidden_dim//2)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                head_i = heads[..., :hidden_dim // 2, :]
                head_j = heads[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, head_i) + th.bmm(backward_tmp, head_j))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                head_i = heads[..., :hidden_dim // 2]
                head_j = heads[..., hidden_dim // 2:]
                rel = relations[..., :hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (head_i * rel).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * head_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tail_i = tails[..., :hidden_dim // 2, :]
                tail_j = tails[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, tail_j) + th.bmm(backward_tmp, tail_i))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn
#                             ----上面是所有对比的kge-model的三元组评分指标----


class KEModel(object):
    """ DGL Knowledge Embedding Model.

    Parameters
    ----------
    args:
        Global configs.
    model_name : str
        Which KG model to use, including 'TransE_l1', 'TransE_l2', 'DistMult', 'ComplEx', 'RotatE', 'SimplE'
    n_entities : int
        Num of entities.
    n_relations : int
        Num of relations.
    hidden_dim : int
        Dimension size of embedding.
    gamma : float
        Gamma for score function.
    double_entity_emb : bool
        If True, entity embedding size will be 2 * hidden_dim.
        Default: False
    double_relation_emb : bool
        If True, relation embedding size will be 2 * hidden_dim.
        Default: False
    """

    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False):
        super(KEModel, self).__init__()
        self.args = args
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / hidden_dim
        device = get_device(args)

        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim  # 都是300
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim
        if model_name == 'RESCAL':
            rel_dim = relation_dim * entity_dim
        else:
            rel_dim = relation_dim
        self.rel_dim = rel_dim
        self.entity_dim = entity_dim
        self.strict_rel_part = args.strict_rel_part
        self.soft_rel_part = args.soft_rel_part

        self.entity_emb = ExternalEmbedding(args, n_entities, entity_dim,
                                            F.cpu() if args.mix_cpu_gpu else device)
        if not self.strict_rel_part and not self.soft_rel_part:
            self.relation_emb = ExternalEmbedding(args, n_relations, rel_dim,       # 如果是RESCAL，这里rel_dim就是90000
                                                  F.cpu() if args.mix_cpu_gpu else device)
        else:
            self.global_relation_emb = ExternalEmbedding(args, n_relations, rel_dim, F.cpu())

        if model_name == 'TransE' or model_name == 'TransE_l2':
            self.score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            self.score_func = TransEScore(gamma, 'l1')
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'ComplEx':
            self.score_func = ComplExScore()
        elif model_name == 'RotatE':
            self.score_func = RotatEScore(gamma, self.emb_init)
        elif model_name == 'SimplE':
            self.score_func = SimplEScore()

        self.model_name = model_name
        self.head_neg_score = self.score_func.create_neg(True)
        self.tail_neg_score = self.score_func.create_neg(False)
        self.head_neg_prepare = self.score_func.create_neg_prepare(True)
        self.tail_neg_prepare = self.score_func.create_neg_prepare(False)

        self.reset_parameters()
        # print(self.entity_emb)

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process embeddings access.
        """
        self.entity_emb.share_memory()
        if self.strict_rel_part or self.soft_rel_part:
            self.global_relation_emb.share_memory()
        else:
            self.relation_emb.share_memory()

        if self.model_name == 'TransR':
            self.score_func.share_memory()

    # 下面两个函数是对模型的保存和加载
    def save_emb(self, path, dataset):
        """Save the model.

        Parameters
        ----------
        path : str
            Directory to save the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.save(path, dataset + '_' + self.model_name + '_entity')
        if self.strict_rel_part or self.soft_rel_part:
            self.global_relation_emb.save(path, dataset + '_' + self.model_name + '_relation')
        else:
            self.relation_emb.save(path, dataset + '_' + self.model_name + '_relation')

        self.score_func.save(path, dataset + '_' + self.model_name)

    def load_emb(self, path, dataset):
        """Load the model.

        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.load(path, dataset + '_' + self.model_name + '_entity')
        self.relation_emb.load(path, dataset + '_' + self.model_name + '_relation')
        self.score_func.load(path, dataset + '_' + self.model_name)

    def reset_parameters(self):
        """Re-initialize the model.
        """
        self.entity_emb.init(self.emb_init)
        self.score_func.reset_parameters()
        if (not self.strict_rel_part) and (not self.soft_rel_part):
            self.relation_emb.init(self.emb_init)
        else:
            self.global_relation_emb.init(self.emb_init)

    # 下面两个函数是用来计算正负样本的最终得分，在该项目中，-(pos_score + neg_score) / 2作为loss
    def predict_score(self, g):
        """Predict the positive score.

        Parameters
        ----------
        g : DGLGraph
            Graph holding positive edges.

        Returns
        -------
        tensor
            The positive score
        """
        self.score_func(g)
        return g.edata['score']

    def predict_neg_score(self, pos_g, neg_g, to_device=None, gpu_id=-1, trace=False,
                          neg_deg_sample=False):
        """Calculate the negative score.

        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        to_device : func
            Function to move data into device.
        gpu_id : int
            Which gpu to move data to.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: False
        neg_deg_sample : bool
            If True, we use the head and tail nodes of the positive edges to
            construct negative edges.
            Default: False

        Returns
        -------
        tensor
            The negative score
        """
        num_chunks = neg_g.num_chunks
        chunk_size = neg_g.chunk_size
        neg_sample_size = neg_g.neg_sample_size
        mask = F.ones((num_chunks, chunk_size * (neg_sample_size + chunk_size)),
                      dtype=F.float32, ctx=F.context(pos_g.ndata['emb']))
        if neg_g.neg_head:
            neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
            neg_head = self.entity_emb(neg_head_ids, gpu_id, trace)
            head_ids, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0:
                tail_ids = to_device(tail_ids, gpu_id)
            tail = pos_g.ndata['emb'][tail_ids]
            rel = pos_g.edata['emb']

            # When we train a batch, we could use the head nodes of the positive edges to
            # construct negative edges. We construct a negative edge between a positive head
            # node and every positive tail node.
            # When we construct negative edges like this, we know there is one positive
            # edge for a positive head node among the negative edges. We need to mask
            # them.
            if neg_deg_sample:
                head = pos_g.ndata['emb'][head_ids]
                head = head.reshape(num_chunks, chunk_size, -1)
                neg_head = neg_head.reshape(num_chunks, neg_sample_size, -1)
                neg_head = F.cat([head, neg_head], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:, 0::(neg_sample_size + 1)] = 0
            neg_head = neg_head.reshape(num_chunks * neg_sample_size, -1)
            neg_head, tail = self.head_neg_prepare(pos_g.edata['id'], num_chunks, neg_head, tail, gpu_id, trace)
            neg_score = self.head_neg_score(neg_head, rel, tail,
                                            num_chunks, chunk_size, neg_sample_size)
        else:
            neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
            neg_tail = self.entity_emb(neg_tail_ids, gpu_id, trace)
            head_ids, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0:
                head_ids = to_device(head_ids, gpu_id)
            head = pos_g.ndata['emb'][head_ids]
            rel = pos_g.edata['emb']

            # This is negative edge construction similar to the above.
            if neg_deg_sample:
                tail = pos_g.ndata['emb'][tail_ids]
                tail = tail.reshape(num_chunks, chunk_size, -1)
                neg_tail = neg_tail.reshape(num_chunks, neg_sample_size, -1)
                neg_tail = F.cat([tail, neg_tail], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:, 0::(neg_sample_size + 1)] = 0
            neg_tail = neg_tail.reshape(num_chunks * neg_sample_size, -1)
            head, neg_tail = self.tail_neg_prepare(pos_g.edata['id'], num_chunks, head, neg_tail, gpu_id, trace)
            neg_score = self.tail_neg_score(head, rel, neg_tail,
                                            num_chunks, chunk_size, neg_sample_size)

        if neg_deg_sample:
            neg_g.neg_sample_size = neg_sample_size
            mask = mask.reshape(num_chunks, chunk_size, neg_sample_size)
            return neg_score * mask
        else:
            return neg_score

    def forward_test(self, pos_g, neg_g, logs, gpu_id=-1):
        """Do the forward and generate ranking results.

        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        logs : List
            Where to put results in.
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, False)
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, False)
        self.score_func.prepare(pos_g, gpu_id, False)

        batch_size = pos_g.number_of_edges()
        pos_scores = self.predict_score(pos_g)
        pos_scores = reshape(logsigmoid(pos_scores), batch_size, -1)

        neg_scores = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
                                            gpu_id=gpu_id, trace=False,
                                            neg_deg_sample=self.args.neg_deg_sample_eval)
        neg_scores = reshape(logsigmoid(neg_scores), batch_size, -1)

        # 会用到， We need to filter the positive edges in the negative graph.
        if self.args.eval_filter:
            filter_bias = reshape(neg_g.edata['bias'], batch_size, -1)
            if gpu_id >= 0:
                filter_bias = cuda(filter_bias, gpu_id)
            # find all indices where it is not false negative sample
            neg_scores += filter_bias

        # To compute the rank of a positive edge among all negative edges,
        # we need to know how many negative edges have higher scores than
        # the positive edge.
        rankings = F.sum(neg_scores >= pos_scores, dim=1) + 1
        rankings = F.asnumpy(rankings)
        for i in range(batch_size):
            ranking = rankings[i]
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0
            })

    # @profile
    def forward(self, pos_g, neg_g, gpu_id=-1):
        """Do the forward.

        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.

        Returns
        -------
        tensor
            loss value
        dict
            loss info
        """

        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, True)       # 调用--call--
        pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)
        self.score_func.prepare(pos_g, gpu_id, True)
        pos_score = self.predict_score(pos_g)
        pos_score = logsigmoid(pos_score)
        if gpu_id >= 0:
            neg_score = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
                                               gpu_id=gpu_id, trace=True,   # 训练时trace为true，测试时为false
                                               neg_deg_sample=self.args.neg_deg_sample)
        else:
            neg_score = self.predict_neg_score(pos_g, neg_g, trace=True,
                                               neg_deg_sample=self.args.neg_deg_sample)

        neg_score = reshape(neg_score, -1, neg_g.neg_sample_size)
        # Adversarial sampling
        if self.args.neg_adversarial_sampling:
            neg_score = F.sum(F.softmax(neg_score * self.args.adversarial_temperature, dim=1).detach()
                              * logsigmoid(-neg_score), dim=1)
        else:
            neg_score = F.mean(logsigmoid(-neg_score), dim=1)

        pos_score = pos_score.mean()
        neg_score = neg_score.mean()
        # compute loss
        loss = -(pos_score + neg_score) / 2

        log = {'pos_loss': - get_scalar(pos_score),
               'neg_loss': - get_scalar(neg_score),
               'loss': get_scalar(loss)}

        if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0:
            coef, nm = self.args.regularization_coef, self.args.regularization_norm
            reg = coef * (norm(self.entity_emb.curr_emb(), nm) + norm(self.relation_emb.curr_emb(), nm))
            log['regularization'] = get_scalar(reg)
            loss = loss + reg

        return loss, log

    def update(self, gpu_id=-1):
        """ Update the embeddings in the model

        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        self.entity_emb.update(gpu_id)
        self.relation_emb.update(gpu_id)
        self.score_func.update(gpu_id)

    # args.strict_rel_part、args.soft_rel_part都是false，所以后面四个函数用不到
    def prepare_relation(self, device=None):
        """ Prepare relation embeddings in multi-process multi-gpu training model.

        device : th.device
            Which device (GPU) to put relation embeddings in.
        """
        self.relation_emb = ExternalEmbedding(self.args, self.n_relations, self.rel_dim, device)
        self.relation_emb.init(self.emb_init)
        if self.model_name == 'TransR':
            local_projection_emb = ExternalEmbedding(self.args, self.n_relations,
                                                     self.entity_dim * self.rel_dim, device)
            self.score_func.prepare_local_emb(local_projection_emb)
            self.score_func.reset_parameters()

    def prepare_cross_rels(self, cross_rels):
        self.relation_emb.setup_cross_rels(cross_rels, self.global_relation_emb)
        if self.model_name == 'TransR':
            self.score_func.prepare_cross_rels(cross_rels)

    def writeback_relation(self, rank=0, rel_parts=None):
        """ Writeback relation embeddings in a specific process to global relation embedding.
        Used in multi-process multi-gpu training model.

        rank : int
            Process id.
        rel_parts : List of tensor
            List of tensor stroing edge types of each partition.
        """
        idx = rel_parts[rank]
        if self.soft_rel_part:
            idx = self.relation_emb.get_noncross_idx(idx)
        self.global_relation_emb.emb[idx] = F.copy_to(self.relation_emb.emb, F.cpu())[idx]
        if self.model_name == 'TransR':
            self.score_func.writeback_local_emb(idx)

    def load_relation(self, device=None):
        """ Sync global relation embeddings into local relation embeddings.
        Used in multi-process multi-gpu training model.

        device : th.device
            Which device (GPU) to put relation embeddings in.
        """
        self.relation_emb = ExternalEmbedding(self.args, self.n_relations, self.rel_dim, device)
        self.relation_emb.emb = F.copy_to(self.global_relation_emb.emb, device)
        if self.model_name == 'TransR':
            local_projection_emb = ExternalEmbedding(self.args, self.n_relations,
                                                     self.entity_dim * self.rel_dim, device)
            self.score_func.load_local_emb(local_projection_emb)

    # 后面两个函数和异步更新有关，因为自己用的只有一个gpu，所以没必要使用
    def create_async_update(self):
        """Set up the async update for entity embedding.
        """
        self.entity_emb.create_async_update()

    def finish_async_update(self):
        """Terminate the async update for entity embedding.
        """
        self.entity_emb.finish_async_update()

    # 因为在train函数中client是None，所以下面这两个也不会用到
    def pull_model(self, client, pos_g, neg_g):
        with th.no_grad():
            entity_id = F.cat(seq=[pos_g.ndata['id'], neg_g.ndata['id']], dim=0)
            relation_id = pos_g.edata['id']
            entity_id = F.tensor(np.unique(F.asnumpy(entity_id)))
            relation_id = F.tensor(np.unique(F.asnumpy(relation_id)))

            l2g = client.get_local2global()
            global_entity_id = l2g[entity_id]

            entity_data = client.pull(name='entity_emb', id_tensor=global_entity_id)
            relation_data = client.pull(name='relation_emb', id_tensor=relation_id)

            self.entity_emb.emb[entity_id] = entity_data
            self.relation_emb.emb[relation_id] = relation_data

    def push_gradient(self, client):
        with th.no_grad():
            l2g = client.get_local2global()
            for entity_id, entity_data in self.entity_emb.trace:
                grad = entity_data.grad.data
                global_entity_id = l2g[entity_id]
                client.push(name='entity_emb', id_tensor=global_entity_id, data_tensor=grad)

            for relation_id, relation_data in self.relation_emb.trace:
                grad = relation_data.grad.data
                client.push(name='relation_emb', id_tensor=relation_id, data_tensor=grad)

        self.entity_emb.trace = []
        self.relation_emb.trace = []


def load_model(args, n_entities, n_relations, ckpt=None):
    if args.model_name == 'RotatE':
        args.double_ent = True
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)     # 后面两个参数在RotatE时需要为true
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model


# ----------------------------------------------------tran & case_study-------------------------------------------------------
def test(args, model, test_samplers, rank=0, mode='Test', queue=None):
    gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))
    with th.no_grad():
        logs = []

        for sampler in test_samplers:
            for pos_g, neg_g in sampler:
                model.forward_test(pos_g, neg_g, logs, gpu_id)

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        if queue is not None:
            queue.put(logs)
        else:
            for k, v in metrics.items():
                print('[{}]{} average {}: {}'.format(rank, mode, k, v))
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()


def train(args, model, train_sampler, valid_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None,
          client=None):
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))
    gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    # 下面三个都用不到
    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    for step in range(0, args.max_step):
        start1 = time.time()
        pos_g, neg_g = next(train_sampler)
        sample_time += time.time() - start1     # 采样时间

        if client is not None:
            model.pull_model(client, pos_g, neg_g)

        start1 = time.time()
        loss, log = model.forward(pos_g, neg_g, gpu_id)
        forward_time += time.time() - start1    # 前反馈时间，也就是计算loss的时间

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1   # 后反馈时间，也就是更新梯度的时间

        start1 = time.time()
        if client is not None:
            model.push_gradient(client)
        else:
            model.update(gpu_id)
        update_time += time.time() - start1
        logs.append(log)

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
                (step + 1) % args.force_sync_interval == 0:
            barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                logs = []
                print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                             time.time() - start))
                print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()

        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            if args.strict_rel_part or args.soft_rel_part:
                model.writeback_relation(rank, rel_parts)
            # forced sync for validation
            if barrier is not None:
                barrier.wait()
            test(args, model, valid_samplers, rank, mode='Valid')
            print('[proc {}]validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            if args.soft_rel_part:
                model.prepare_cross_rels(cross_rels)
            if barrier is not None:
                barrier.wait()

    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
    if args.async_update:
        model.finish_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)


# ----------------------------------------------------main 函数----------------------------------------------------------
def get_emb(args):
    prepare_save_path(args)
    init_time_start = time.time()

    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files,
                          args.has_edge_importance)

    g = ConstructGraph(dataset, args)  # 这里其实会把train\valid\test做一个合并，生成一个图
    train_data = TrainDataset(g, dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance)
    args.eval_filter = not args.no_eval_filter  # eval_filter=True，模型在评估时会过滤掉一些不合适的负样本
    args.num_workers = 8

    args.soft_rel_part = args.mix_cpu_gpu and args.rel_part  # False
    args.strict_rel_part = args.mix_cpu_gpu and (train_data.cross_part == False)  # False
    # 负采样，随机替代头部和随机替代尾部
    train_sampler_head = train_data.create_sampler(args.batch_size,  # 2048
                                                   args.neg_sample_size,  # 128
                                                   args.neg_sample_size,  # 128
                                                   mode='head',
                                                   num_workers=args.num_workers,  # 8
                                                   shuffle=True,
                                                   exclude_positive=False)
    train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                   args.neg_sample_size,
                                                   args.neg_sample_size,
                                                   mode='tail',
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   exclude_positive=False)
    train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                    args.neg_sample_size, args.neg_sample_size,
                                                    True, dataset.n_entities,
                                                    args.has_edge_importance)

    rel_parts = train_data.rel_parts if args.strict_rel_part or args.soft_rel_part else None  # None
    cross_rels = train_data.cross_rels if args.soft_rel_part else None  # None
    gc.collect()  # 释放内存

    if args.valid or args.test:
        args.num_test_proc = args.num_proc
        if args.valid:
            assert dataset.valid is not None, 'validation set is not provided'
        if args.test:
            assert dataset.test is not None, 'case_study set is not provided'
        eval_dataset = EvalDataset(g, dataset, args)

    if args.test:
        test_sampler_head = eval_dataset.create_sampler('case_study', args.batch_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.eval_filter,
                                                        mode='chunk-head',
                                                        num_workers=args.num_workers,
                                                        rank=0, ranks=1)
        test_sampler_tail = eval_dataset.create_sampler('case_study', args.batch_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.eval_filter,
                                                        mode='chunk-tail',
                                                        num_workers=args.num_workers,
                                                        rank=0, ranks=1)

    # load model
    n_entities = dataset.n_entities  # 133195
    n_relations = dataset.n_relations  # 29
    emap_file = dataset.emap_fname
    rmap_file = dataset.rmap_fname
    gc.collect()

    model = load_model(args, n_entities, n_relations)
    if args.num_proc > 1 or args.async_update:  # False
        model.share_memory()

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

    # train
    start = time.time()
    valid_samplers = None
    train(args, model, train_sampler, valid_samplers, rel_parts=rel_parts)

    print('training takes {} seconds'.format(time.time() - start))

    if not args.no_save_emb:
        save_model(args, model, emap_file, rmap_file)

    # case_study
    if args.test:
        start = time.time()
        test(args, model, [test_sampler_head, test_sampler_tail])
        print('testing takes {:.3f} seconds'.format(time.time() - start))
    return os.path.join(args.save_path, 'primeKG_{}_entity.npy'.format(args.model_name))


if __name__ == '__main__':
    args = ArgParser().parse_args()
    entity_emb_path = get_emb(args)

    # 后面代码的目的是补全所有的drug的embedding,并保存
    entity2emb = {}
    kg_drug_emb = []
    entity_file = '../data_after_processing/kg/primeKG/entities.tsv'
    approved_drug_file = '../data/approved_structure links.csv'
    kg_drug_emb_file = 'embedding/kg_{}_drug_embeddings.npy'.format(args.model_name)

    with open(entity_file, 'r', encoding='utf-8') as f:
        entity_emb = np.load(entity_emb_path)
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            entity2emb[row[1]] = entity_emb[int(row[0])].reshape((1, 300))

    with open(approved_drug_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            name = row[1]
            if name in entity2emb.keys():
                emb = entity2emb[name]
            else:
                emb = np.zeros((1, 300))
            kg_drug_emb.append(emb)
    kg_drug_emb = np.concatenate(kg_drug_emb, axis=0)
    np.save(kg_drug_emb_file, kg_drug_emb)
