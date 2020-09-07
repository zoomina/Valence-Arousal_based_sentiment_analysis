import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time

class GetData:
    def __init__(self, lang, col, handle_data='drop', jj=0.1, ad=0.1, nn=0.1):
        self.lang = lang
        self.col = col
        self.clen = len(col)
        self.handle_data = handle_data
        self.jj = jj
        self.ad = ad
        self.nn = nn

    def load_data(self, path):
        data = pd.read_csv(path, dtype={c:np.float64 for c in self.col})
        data[self.lang] = data.apply(lambda row: eval(row[self.lang]), axis=1)

        if self.handle_data == 'drop':
            return self.to_nparray_drop(data[self.lang], data[self.col])
        elif self.handle_data == 'fill':
            return self.to_nparray_fill(data[self.lang], data[self.col])
        elif self.handle_data == 'not_split':
            return data

    def _padding(self, np_X, y):
        test_idx = np.random.choice(len(np_X), len(np_X) // 5, replace=False)
        # train_idx = [i for i in range(np_X.shape[0]) if i not in test_idx]
        train_idx = list(set(range(len(np_X))) - set(test_idx))

        # find max len (from all sentences)
        max_len = 0
        for row in np_X:
            _r = row.shape[0]
            if max_len < _r:
                max_len = _r
        print(max_len)

        # 0 padding
        pad = np.full((len(np_X), max_len, self.clen), 0, dtype=np.float64)

        for i, row in enumerate(np_X):
            if row.shape == (0, ):
                continue
            pad[i, :row.shape[0]] = row

        train_X = np.array(pad[train_idx])
        train_y = np.array(y, dtype=np.float64)[train_idx]
        test_X = np.array(pad[test_idx])
        test_y = np.array(y, dtype=np.float64)[test_idx]

        return train_X, train_y, test_X, test_y

    def pos_weight(self, s):
        for k, v in s.items():
            v = np.array(v)
            if len(k) == 1:
                s[k] = v*self.jj
            elif k[1] in ['RB', 'RBR', 'RBS']:
                s[k] = v*self.ad
            elif k[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'PTD', 'POS', 'PRP', 'PRP$']:
                s[k] = v*self.nn
            elif k[1] not in ['JJ', 'JJR', 'JJS']:
                s[k] = v*self.jj

        return s

    def to_nparray_fill(self, X, y):
        np_X = []
        for i, s in enumerate(X):
            clean_s = {k: v if v is not None else [0]*self.clen for k, v in s.items()}
            clean_s = self.pos_weight(clean_s)
            np_X.append(np.array([v for v in clean_s.values()]))

        return self._padding(np_X, y)

    def to_nparray_drop(self, X, y):
        np_X = []
        for i, s in enumerate(X):
            clean_s = {k: v for k, v in s.items() if v is not None}
            clean_s = self.pos_weight(clean_s)
            np_X.append(np.array([v for v in clean_s.values()]))

        return self._padding(np_X, y)

def justCount(X, y, clen):
    predict = []
    for i, s in enumerate(X):
        # clean_s : <class 'dict'>, word:[v, a]
        clean_s = {k: v for k, v in s.items() if v is not None}
        s_val = [v for v in clean_s.values()]

        try:
            predict.append(sum(s_val) / len(clean_s))
        except:
            predict.append([0]*clen)

    mse = mean_squared_error(y, predict)
    print("just count:", mse)

class LSTM(nn.Module):
    def __init__(self, i_size, h_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(i_size, h_size, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2*h_size, 2)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.linear(x)
        return x

class UsingLSTM:
    def __init__(self, col):
        self.batch_size = 64
        self.i_size, self.h_size = len(col), len(col)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = nn.MSELoss()

    def build_model(self):
        model = nn.Sequential(nn.LSTM(self.i_size, self.h_size, batch_first=True, bidirectional=False))
        return model.to(self.device)

    def train(self, X, y, iter_size=50):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        scheduler = StepLR(self.adam, step_size=10, gamma=0.1)
        self.model.train()
        self.model.train()
        train_summary = {"loss": []}
        for i in range(iter_size):
            start_time = time.time()
            step_size = X.shape[0]//self.batch_size
            step_summary = {"loss" : 0}
            for step in range(step_size):
                batch_mask = range(i * self.batch_size, (i + 1) * self.batch_size)
                self.adam.zero_grad()  # zero the gradient buffers
                output, _ = self.model(X[batch_mask])
                loss = self.loss(output[:,-1,:], y[batch_mask])
                loss.backward()
                self.adam.step()  # Does the update
                step_summary["loss"] += loss.data

            scheduler.step()
            print("epoch [{:3}/{:3}] time [{:6.4f}s] loss [{:.7f}]".format(i+1, iter_size, time.time()-start_time, step_summary["loss"]/step_size))
            train_summary["loss"].append(step_summary["loss"]/step_size)

        plt.plot(range(len(train_summary["loss"])), train_summary["loss"])
        plt.show()

    def test(self, X, y):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            test_summary = {"loss": []}
            for i in range(X.shape[0]):
                output, _ = self.model(X[i].unsqueeze(dim=0))
                loss = self.loss(output[:, -1, :], y[i])
                test_summary["loss"].append(loss.detach().cpu().numpy())

            print("Test complete : avg. loss :", sum(test_summary["loss"])/len(test_summary["loss"]))

class UsingBiLSTM:
    def __init__(self, col):
        self.batch_size = 32
        self.i_size, self.h_size = len(col), len(col)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.loss = nn.MSELoss()

    def build_model(self):
        model = LSTM(self.i_size, self.h_size)
        return model.to(self.device)

    def train(self, X, y, iter_size=100):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        scheduler = StepLR(self.adam, step_size=20, gamma=0.1)
        self.model.train()
        self.model.train()
        train_summary = {"loss": []}
        for i in range(iter_size):
            start_time = time.time()
            step_size = X.shape[0]//self.batch_size
            step_summary = {"loss" : 0}
            for step in range(step_size):
                batch_mask = range(i * self.batch_size, (i + 1) * self.batch_size)
                self.adam.zero_grad()  # zero the gradient buffers
                output = self.model(X[batch_mask])
                loss = self.loss(output, y[batch_mask])
                loss.backward()
                self.adam.step()  # Does the update
                step_summary["loss"] += loss.data

            scheduler.step()
            print("epoch [{:3}/{:3}] time [{:6.4f}s] loss [{:.7f}]".format(i+1, iter_size, time.time()-start_time, step_summary["loss"]/step_size))
            train_summary["loss"].append(step_summary["loss"]/step_size)

        plt.plot(range(len(train_summary["loss"])), train_summary["loss"])
        plt.show()

    def test(self, X, y):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            test_summary = {"loss": []}
            for i in range(X.shape[0]):
                output = self.model(X[i].unsqueeze(dim=0))
                loss = self.loss(output, y[i])
                test_summary["loss"].append(loss.detach().cpu().numpy())

            print("Test complete : avg. loss :", sum(test_summary["loss"])/len(test_summary["loss"]))

class CNN(nn.Module):
    def __init__(self, i_size, h_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(i_size, 8, kernel_size=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.adapool = nn.AdaptiveAvgPool1d(1)
        self.lin1 = nn.Linear(32, 16)
        self.lin2 = nn.Linear(16, h_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))
        x = self.adapool(x)
        x = x.flatten(1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

class UsingCNN:
    def __init__(self, col):
        self.batch_size = 64
        self.i_size, self.h_size = len(col), len(col)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = nn.MSELoss()

    def build_model(self):
        model = CNN(self.i_size, self.h_size)
        return model.to(self.device)

    def train(self, X, y, iter_size=50):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        scheduler = StepLR(self.adam, step_size=10, gamma=0.1)
        self.model.train()
        train_summary = {"loss": []}
        for i in range(iter_size):
            start_time = time.time()
            step_size = X.shape[0]//self.batch_size
            step_summary = {"loss" : 0}
            for step in range(step_size):
                batch_mask = range(i * self.batch_size, (i + 1) * self.batch_size)
                self.adam.zero_grad()  # zero the gradient buffers
                output = self.model(X[batch_mask])
                loss = self.loss(output, y[batch_mask])
                loss.backward()
                self.adam.step()  # Does the update
                step_summary["loss"] += loss.data

            scheduler.step()
            print("epoch [{:3}/{:3}] time [{:6.4f}s] loss [{:.7f}]".format(i+1, iter_size, time.time()-start_time, step_summary["loss"]/step_size))
            train_summary["loss"].append(step_summary["loss"]/step_size)

        plt.plot(range(len(train_summary["loss"])), train_summary["loss"])
        plt.show()

    def test(self, X, y):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            test_summary = {"loss": [], "output": []}
            for i in range(X.shape[0]):
                output = self.model(X[i].unsqueeze(dim=0))
                loss = self.loss(output, y[i])
                test_summary["loss"].append(loss.detach().cpu().numpy())

            print("Test complete : avg. loss :", sum(test_summary["loss"])/len(test_summary["loss"]))

class CLSTM(nn.Module):
    def __init__(self, i_size, h_size):
        super(CLSTM, self).__init__()
        self.conv1 = nn.Conv1d(i_size, 8, kernel_size=7)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.adapool = nn.AdaptiveAvgPool1d(4)
        self.lstm = nn.LSTM(4, 4, batch_first=True, bidirectional=False)
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, h_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))
        x = self.adapool(x)
        x, _ = self.lstm(x)
        x = x.flatten(1)
        x = self.lin1(x)
        x = self.lin2(x)

        return x

class UsingCLSTM:
    def __init__(self, col):
        self.batch_size = 64
        self.i_size, self.h_size = len(col), len(col)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e-7, weight_decay=1e-4)
        self.loss = nn.MSELoss()

    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d)):
            try:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            except:
                pass

    def build_model(self):
        model = CLSTM(self.i_size, self.h_size)
        model.apply(self.init_weights)
        return model.to(self.device)

    def train(self, X, y, iter_size=100):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        scheduler = StepLR(self.adam, step_size=20, gamma=0.1)
        self.model.train()
        self.model.train()
        train_summary = {"loss": []}
        for i in range(iter_size):
            start_time = time.time()
            step_size = X.shape[0] // self.batch_size
            step_summary = {"loss": 0}
            for step in range(step_size):
                batch_mask = range(i * self.batch_size, (i + 1) * self.batch_size)
                self.adam.zero_grad()  # zero the gradient buffers
                output = self.model(X[batch_mask])
                loss = self.loss(output, y[batch_mask])
                loss.backward()
                self.adam.step()  # Does the update
                step_summary["loss"] += loss.data

            scheduler.step()
            print(
                "epoch [{:3}/{:3}] time [{:6.4f}s] loss [{:.7f}]".format(i + 1, iter_size, time.time() - start_time,
                                                                         step_summary["loss"] / step_size))
            train_summary["loss"].append(step_summary["loss"] / step_size)

        plt.plot(range(len(train_summary["loss"])), train_summary["loss"])
        plt.show()

    def test(self, X, y):
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y = torch.tensor(y, dtype=torch.float).to(self.device)

        self.model.eval()
        with torch.no_grad():
            test_summary = {"loss": []}
            for i in range(X.shape[0]):
                output = self.model(X[i].unsqueeze(dim=0))
                loss = self.loss(output, y[i])
                test_summary["loss"].append(loss.detach().cpu().numpy())

            print("Test complete : avg. loss :", sum(test_summary["loss"]) / len(test_summary["loss"]))

def main():

    lang = 'ENG'
    col = ['V', 'A']
    path = 'data/preprocessed_ML.csv'

    train_X, train_y, test_X, test_y = GetData(lang, col, handle_data='fill', jj=1, ad=1, nn=1).load_data(path)
    print('==== CNN ====')
    cnn = UsingCNN(col)
    cnn.train(train_X, train_y)
    cnn.test(test_X, test_y)

    print('==== BiLSTM ====')
    lstm = UsingBiLSTM(col)
    lstm.train(train_X, train_y)
    lstm.test(test_X, test_y)

    print('==== CLSTM ====')
    clstm = UsingCLSTM(col)
    clstm.train(train_X, train_y)
    clstm.test(test_X, test_y)

    train_X, train_y, test_X, test_y = GetData(lang, col, handle_data='fill').load_data(path)
    print('==== LSTM ====')
    lstm = UsingLSTM(col)
    lstm.train(train_X, train_y)
    lstm.test(test_X, test_y)

    data = GetData(lang, col, 'not_split').load_data(path)
    print('==== Just Count ====')
    print('mse: ', end='')
    justCount(data[lang], data[col], len(col))


if __name__ == '__main__':
    main()
