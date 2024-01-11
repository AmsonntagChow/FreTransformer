from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import os

# traffic/Solar/Wiki data
class Dataset_Dhfm(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        load_data = np.load(root_path)
        data = load_data.transpose()
        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data)*0.7)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data)*0.7)
            end = int(len(data)*0.9)
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data)*0.9)
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        return data, next_data

    def __len__(self):
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len

# ECG/COVID/Exchange/METR-LA/Electricity dataset
class Dataset_ECG(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path)

        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        data = np.array(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data)*self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data)*self.train_ratio)
            end = int(len(data)*(self.val_ratio+self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data)*(self.val_ratio+self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len

# Financial dataset
class Dataset_Fin(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, start_train, end_train, start_vali, end_vali, start_backtest, end_backtest):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.start_train = pd.to_datetime(start_train)
        self.end_train = pd.to_datetime(end_train)
        self.start_vali = pd.to_datetime(start_vali)
        self.end_vali = pd.to_datetime(end_vali)
        self.start_backtest = pd.to_datetime(start_backtest)
        self.end_backtest = pd.to_datetime(end_backtest)

        data = pd.read_csv(root_path)

        if type == '1':
            self.mms = MinMaxScaler(feature_range=(0, 1))
            data_values = self.mms.fit_transform(data.iloc[:, 1:])
        else:
            data_values = data.values[:,1:]

        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])

        data_values = np.array(data_values)
        data = np.array(data)


        if self.flag == 'train':
            mask = (data[:, 0] >= self.start_train) & (data[:, 0] <= self.end_train)
            self.trainData = data_values[mask]
        if self.flag == 'val':
            mask = (data[:, 0] >= self.start_vali) & (data[:, 0] <= self.end_vali)
            self.valData = data_values[mask]
        if self.flag == 'test':
            mask = (data[:, 0] >= self.start_backtest) & (data[:, 0] <= self.end_backtest)
            self.testData = data_values[mask]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len
    
    def inverse(self):
        return self.mms.inverse_transform(self.data)
    
# Optiver dataset
class Dataset_Opt(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # if type == '1':
        #     mms = MinMaxScaler(feature_range=(0, 1))
        #     data = mms.fit_transform(data)
        # data = np.array(data)


        if self.flag == 'train':
            file_name = 'train.csv'
            full_path = os.path.join(self.path, file_name)
            data = pd.read_csv(full_path)
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
            data = np.array(data)
            begin = 0
            end = int(len(data)*self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            file_name = 'train.csv'
            full_path = os.path.join(self.path, file_name)
            data = pd.read_csv(full_path)
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
            data = np.array(data)
            begin = int(len(data)*self.train_ratio)
            end = len(data)
            self.valnData = data[begin:end]
        if self.flag == 'test':
            file_name = 'train.csv'
            full_path = os.path.join(self.path, file_name)
            data = pd.read_csv(full_path)
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
            data = np.array(data)
            self.testData = data

        

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len