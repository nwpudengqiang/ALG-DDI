from sklearn.model_selection import KFold, StratifiedKFold
import csv
import numpy as np


# load data
class DataLoader(object):

    def __init__(self, args):
        self.args = args
        # embedding 都应该是2715*300
        self.structure_embed = np.load('embedding/st_embedding.npy')
        self.drug_protein_embed = np.load('embedding/protein_drug_embeddings.npy')
        self.drug_effect_embed = np.load('embedding/effect_drug_embeddings.npy')
        self.drug_disease_embed = np.load('embedding/disease_drug_embeddings.npy')
        self.kg_embed = np.load('embedding/kg_{}_drug_embeddings.npy'.format(args.KGE_model))
        if args.ddi_dataset == 'DS1':
            self.ddi_file = 'data_after_processing/predictor/DDI_pos_neg.txt'
        if args.ddi_dataset == 'DS2':
            self.ddi_file = 'data_after_processing/predictor/v5_DDI_pos_neg.txt'
        if args.ddi_dataset == 'DS3':
            self.ddi_file = 'data_after_processing/predictor/v4_DDI_pos_neg.txt'
        if args.ddi_dataset == 'DS1_without_case_drug':
            self.ddi_file = 'data_after_processing/case_study/{}_train_ddi.txt'.format(args.case_drug)
        self.multi_ddi_file = 'data_after_processing/predictor/multi_DDI_pos_neg.tsv'

        self.DDI_train_data_X, self.DDI_train_data_Y, self.DDI_test_data_X, self.DDI_test_data_Y = self.load_DDI_data()
        self.n_ddi_train, self.n_ddi_test = self.statistic_ddi_data()  # 计算每个训练集里面的数量

    def load_DDI_data(self):
        if not self.args.multi_class:
            ddi_file = open(self.ddi_file, 'r')
            DDI_id_pair = []
            label = []
            for line in ddi_file.readlines():
                row = line.strip().split('\t')
                DDI_left_id = int(row[0])
                DDI_right_id = int(row[1])
                DDI_id_pair.append([DDI_left_id, DDI_right_id])
                label.append(int(row[2]))
            DDI_id_pair = np.array(DDI_id_pair)
            label = np.array(label)
            kfold = KFold(n_splits=self.args.n_fold, shuffle=True, random_state=3)  # 三折交叉验证，并且打乱，然后得到训练集和测试集
        else:
            DDI_id_pair = []
            label = []
            with open(self.multi_ddi_file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    DDI_left_id = int(row[0])
                    DDI_right_id = int(row[1])
                    DDI_id_pair.append([DDI_left_id, DDI_right_id])
                    label.append(int(row[2]))
            DDI_id_pair = np.array(DDI_id_pair)
            label = np.array(label)
            kfold = StratifiedKFold(n_splits=self.args.n_fold, shuffle=True, random_state=3)     # 测试集里面都有82种类别

        train_X_data = []
        train_Y_data = []
        test_X_data = []
        test_Y_data = []
        for train, test in kfold.split(DDI_id_pair, label):
            train_X_data.append(DDI_id_pair[train])
            train_Y_data.append(label[train])
            test_X_data.append(DDI_id_pair[test])
            test_Y_data.append(label[test])
        print('Loading DDI data down!')
        return train_X_data, train_Y_data, test_X_data, test_Y_data  # 返还训练集和测试集

    # 得到每一个训练集里面的DDI数量
    def statistic_ddi_data(self):
        data_train = []
        for i in range(len(self.DDI_train_data_X)):
            data_train.append(len(self.DDI_train_data_X[i]))
        data_test = []
        for i in range(len(self.DDI_test_data_Y)):
            data_test.append(len(self.DDI_test_data_Y[i]))
        return data_train, data_test