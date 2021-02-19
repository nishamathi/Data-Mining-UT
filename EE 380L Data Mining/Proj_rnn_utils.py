import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import time
import pickle as pkl


def save_model(model, save_dir, hyperparam_dict):
    filename = model.__class__.__name__
    for key, value in hyperparam_dict.items():
        filename = filename + '_' + key + '_' + str(value)

    t = time.localtime()
    filename = filename + '_' + str(t.tm_mday) + '_' + str(t.tm_mon) + '_' + str(t.tm_hour) + '_' + str(t.tm_min) + '.pkl'
    save_path = os.path.join(save_dir, filename)

    model = model.cpu()
    if 'tensors_to_cpu' in dir(model):
        model.tensors_to_cpu()

    pkl.dump(model, open(save_path, 'wb'))

    print("Model written to path: " + save_path)


class TargetEncoder():
    def __init__(self):
        self.category_maps = {}
        return

    def keys(self):
        return self.category_maps.keys()

    def fit(self, X, y, keys):
        if type(keys) != list:
            keys = [keys]

        for key in keys:
            print("Fitting column {}".format(key))
            category_map = {}
            for category, group in X.groupby(key, as_index=False):
                category_map[category] = y.loc[y.index.isin(group.index)].mean()
            self.category_maps[key] = category_map

    def transform(self, X):
        retX = X.copy()
        for key in self.keys():
            retX[key] = retX[key].map(self.category_maps[key])
        
        return retX


class DataGenerator(object):
    def __init__(self, path, batch_size=1, mode='train', use_cuda=True):
        assert mode in ['train', 'test'], 'Invalid mode. Must be "train" or "test"'

        self.path = path
        self.batch_size = batch_size
        self.mode = mode
        self.use_cuda = use_cuda

        df = pd.read_csv(path)
        df.drop(columns=['Unnamed: 0', 'unitdischargeoffset', 'uniquepid', 'hospitaldischargestatus', 'unitdischargestatus'], inplace=True)
        df.set_index('patientunitstayid', inplace=True)
        self.y = df['rlos']
        self.X = df.drop(columns=['rlos'])
        del df

        self.stayids = self.X.index.unique()
        train_ids, test_ids = train_test_split(self.stayids, test_size=0.2, random_state=0)
        self.stayids = train_ids if mode == 'train' else test_ids
        self.n_ids = len(self.stayids)

        self.X = self.X.loc[self.stayids]
        self.y = self.y.loc[self.stayids]

        if not os.path.exists('models'):
            os.mkdir('models')

        encoder_path = os.path.join('models', 'targetencoder.pkl')
        if os.path.exists(encoder_path):
            encoder = pkl.load(open(encoder_path, 'rb'))
        else:
            if mode == 'test':
                print('Encoder not found')
                return
            
            print('Fitting TargetEncoder')
            encoder = TargetEncoder()
            encoder.fit(self.X, self.y, ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
            pkl.dump(encoder, open(encoder_path, 'wb'))

        print('Transforming using TargetEncoder')
        self.X = encoder.transform(self.X)

        scaler_path = os.path.join('models', 'minmaxscaler.pkl')
        if os.path.exists(scaler_path):
            scaler = pkl.load(open(scaler_path, 'rb'))
        else:
            if mode == 'test':
                print('Scaler not found')
                return

            print('Fitting MinMaxScaler')
            scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
            scaler.fit(self.X)
            pkl.dump(scaler, open(scaler_path, 'wb'))

        print('Transforming using MinMaxScaler')
        self.X[self.X.keys()] = scaler.transform(self.X)
        self.steps_per_epoch = self.n_ids//self.batch_size

        self.shuffle()

    def shuffle(self):
        self.inds = np.random.permutation(self.n_ids)
        self.step = 0

    def next(self):
        if self.step == self.steps_per_epoch:
            self.shuffle()

        ids = self.stayids[self.inds[self.step*self.batch_size: (self.step+1)*self.batch_size]]

        xs = []
        ys = []
        for train_id in ids:
            temp_x = self.X.loc[train_id].copy()
            temp_x = torch.from_numpy(temp_x.values).unsqueeze(0).float()
            temp_y = self.y.loc[train_id].copy()
            if len(temp_x.shape) == 2:
                temp_x = temp_x.unsqueeze(0)
                temp_y = torch.tensor([temp_y]).unsqueeze(0).unsqueeze(0).float()
            else:
                temp_y = torch.from_numpy(temp_y.values).unsqueeze(0).float()
            if self.use_cuda:
                temp_x = temp_x.cuda()
                temp_y = temp_y.cuda()
            xs.append(temp_x)
            ys.append(temp_y)

        self.step += 1
        return (xs, ys)


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_recurrent_layers=3):
        super(BiLSTMModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_recurrent_layers = n_recurrent_layers

        self.root_map = nn.Conv1d(input_size, embedding_size, kernel_size=1)
        self.bilstm_cell = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True, bidirectional=True)
        self.predictor = nn.Conv1d(hidden_size*2, 1, kernel_size=1)
    
        self.h_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))
        self.c_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))

    def tensors_to_cuda(self):
        self.h_init = self.h_init.cuda()
        self.c_init = self.c_init.cuda()

    def tensors_to_cpu(self):
        self.h_init = self.h_init.cpu()
        self.c_init = self.c_init.cpu()

    def forward(self, x):
        assert x.size(0) == 1, 'Only one example can be processed at once'

        # Initialize hidden layers
        self.h_init.fill_(0.0)
        self.c_init.fill_(0.0)

        x = x.permute(0, 2, 1) # Interpret features as channels for 1D convolution
        v = self.root_map(x)
        v = v.permute(0, 2, 1) # Move features back to last axis, for LSTM layer

        z, _ = self.bilstm_cell(v, (self.h_init, self.c_init))
        z = z.permute(0, 2, 1) # Interpret features as channels for 1D convolution
        
        y = nn.ReLU()(self.predictor(z).squeeze(1)) # Reshape to 1 x seq_length

        return y


class LSTMModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_recurrent_layers=3):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_recurrent_layers = n_recurrent_layers

        self.root_map = nn.Conv1d(input_size, embedding_size, kernel_size=1)
        self.lstm_cell = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True)
        self.predictor = nn.Conv1d(hidden_size, 1, kernel_size=1)
    
        self.h_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.c_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))

    def tensors_to_cuda(self):
        self.h_init = self.h_init.cuda()
        self.c_init = self.c_init.cuda()

    def tensors_to_cpu(self):
        self.h_init = self.h_init.cpu()
        self.c_init = self.c_init.cpu()

    def forward(self, x):
        assert x.size(0) == 1, 'Only one example can be processed at once'

        # Initialize hidden layers
        self.h_init.fill_(0.0)
        self.c_init.fill_(0.0)

        x = x.permute(0, 2, 1) # Interpret features as channels for 1D convolution
        v = self.root_map(x)
        v = v.permute(0, 2, 1) # Move features back to last axis, for LSTM layer

        z, _ = self.lstm_cell(v, (self.h_init, self.c_init))
        z = z.permute(0, 2, 1) # Interpret features as channels for 1D convolution
        
        y = nn.ReLU()(self.predictor(z).squeeze(1)) # Reshape to 1 x seq_length

        return y


class RETAINModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_recurrent_layers=3):
        super(RETAINModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_recurrent_layers = n_recurrent_layers

        self.root_map = nn.Conv1d(input_size, embedding_size, kernel_size=1)
        self.visit_attention_lstm = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True)
        self.visit_attention_map = nn.Conv1d(hidden_size, 1, kernel_size=1)
        self.feature_attention_lstm = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True)
        self.feature_attention_map = nn.Conv1d(hidden_size, embedding_size, kernel_size=1)
        self.predictor = nn.Conv1d(embedding_size, 1, kernel_size=1)
    
        self.alpha_h_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.alpha_c_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.beta_h_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.beta_c_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))

    def tensors_to_cuda(self):
        self.alpha_h_init = self.alpha_h_init.cuda()
        self.alpha_c_init = self.alpha_c_init.cuda()
        self.beta_h_init = self.beta_h_init.cuda()
        self.beta_c_init = self.beta_c_init.cuda()

    def tensors_to_cpu(self):
        self.alpha_h_init = self.alpha_h_init.cpu()
        self.alpha_c_init = self.alpha_c_init.cpu()
        self.beta_h_init = self.beta_h_init.cpu()
        self.beta_c_init = self.beta_c_init.cpu()

    def forward(self, x):
        assert x.size(0) == 1, 'Only one example can be processed at once'

        # Reverse order of events
        x = x[:, ::-1, :]

        # Initialize hidden layers
        self.alpha_h_init.fill_(0.0)
        self.alpha_c_init.fill_(0.0)
        self.beta_h_init.fill_(0.0)
        self.beta_c_init.fill_(0.0)
        
        x = x.permute(0, 2, 1) # Interpret features as channels for 1D convolution
        v = self.root_map(x)
        v = v.permute(0, 2, 1) # Move features back to last axis, for LSTM layer

        alpha_z, _ = self.visit_attention_lstm(v, (self.alpha_h_init, self.alpha_c_init))
        beta_z, _ = self.feature_attention_lstm(v, (self.beta_h_init, self.beta_c_init))
        alpha_z = alpha_z.permute(0, 2, 1) # Interpret features as channels for 1D convolution
        beta_z = beta_z.permute(0, 2, 1) # Interpret features as channels for 1D convolution
        
        alpha = nn.Sigmoid()(self.visit_attention_map(alpha_z))
        beta = nn.Sigmoid()(self.feature_attention_map(beta_z))

        v = v.permute(0, 2, 1) # Make v compatible with 1D convolution again
        v_weighted = (v * beta) * alpha.repeat(1, self.embedding_size, 1)
        y = nn.ReLU()(self.predictor(z).squeeze(1)) # Reshape to 1 x seq_length

        return y, alpha, beta

