import time

import numpy as np
import torch
import torch.nn as nn
from data_utils import transformer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size,device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class GRU:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    look_forward = None
    n_layers = None
    epochs = None
    batch_size = None
    look_backward = None
    train_loader = None

    def __init__(self,**dict_data):
        #获取完整的请求json数据
        json_dict = dict_data.get('json')
        if json_dict is None:
            return
        #公共参数初始化
        self.look_forward = json_dict.get('lookForward')
        self.look_backward = json_dict.get('lookBackward')
        #超参数初始化
        for key, value in json_dict.get('modelAttr').items():
            if hasattr(self,key):
                #由于传递过来的是一个string-string的map,故需要进行类型转换
                setattr(self, key, int(value))

    def prepare_train_data(self,metrics):
        """
            :param metrics: list of float
        """
        norm_trans = transformer.data_transformer(metrics)
        metrics = norm_trans.normalize()
        # 如修改标签维数，需再进行一次滑动窗口
        print("start preparing data for training")
        metrics = np.array(metrics)

        # 去掉标签数据用不到的部分
        labels_t = metrics[self.look_backward:]
        # 因为标签只有一个数，所以去掉最后一个输入用不上的数
        inputs_t = metrics[:-self.look_forward]

        labels = np.array([])
        for i in range(len(labels_t) - self.look_forward + 1):
            element = labels_t[i:i + self.look_forward]
            labels = np.append(labels, element)

        inputs = np.array([])
        # 滑动窗口，窗口大小为look_back
        for i in range(len(inputs_t) - self.look_backward + 1):
            element = metrics[i:i + self.look_backward]
            # 这个append会把元素全部展开进行append
            inputs = np.append(inputs, element)
        # [[[1.],  [2.],  [3.]],, [[2.],  [3.],  [4.]],, [[3.],  [4.],  [5.]]]
        inputs = inputs.reshape((-1, self.look_backward, 1))
        # [[4,5], [5,6], [6,7]]
        labels = labels.reshape((-1, self.look_forward))

        train_data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, drop_last=True)
        print('training data prepared')
        return self.train_loader

    def train(self,metric_type, learn_rate=0.001, hidden_dim=256):
        # Setting common hyperparameters
        global avg_loss
        input_dim = next(iter(self.train_loader))[0].shape[2]
        output_dim = self.look_forward
        n_layers = self.n_layers
        print('input dim :',input_dim,'output_dim :',output_dim,'n_layers :',n_layers)
        # Instantiating the models
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
        model.to(self.device)

        # Defining loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

        model.train()
        print("Starting Training of GRU model")
        epoch_times = []
        # Start training loop
        for epoch in range(1, self.epochs + 1):
            start_time = time.perf_counter()
            h = model.init_hidden(self.batch_size,self.device)
            avg_loss = 0.
            counter = 0
            for x, label in self.train_loader:
                counter += 1
                h = h.data
                model.zero_grad()

                out, h = model(x.to(self.device).float(), h)
                loss = criterion(out, label.to(self.device).float())
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                if counter % 200 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                               len(self.train_loader),
                                                                                               avg_loss / counter))
            current_time = time.perf_counter()
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, self.epochs, avg_loss / len(self.train_loader)))
            print("Time Elapsed for Epoch: {} seconds".format(str(current_time - start_time)))
            epoch_times.append(current_time - start_time)
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
        torch.save(model.state_dict(), '{}.pt'.format(metric_type))
        return avg_loss / len(self.train_loader)#model

    def predict(self,metrics, metric_type, hidden_dim=256):
        norm_trans = transformer.data_transformer(metrics)
        metrics = norm_trans.normalize()
        metrics = np.array(metrics)
        # 只要后look back个的数据
        metrics = metrics[-self.look_backward:]
        metrics = metrics.reshape((-1, len(metrics), 1))
        input_dim = 1
        output_dim = self.look_forward
        n_layers = self.n_layers
        # Instantiating the models
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
        model.to(self.device)
        model.load_state_dict(torch.load('{}.pt'.format(metric_type)))
        h = model.init_hidden(metrics.shape[0],self.device)
        out, _ = model(torch.from_numpy(metrics).to(self.device).float(), h)
        result = out.tolist()[0]
        return norm_trans.denormalize(result)

class GRUBuilder:
    def __init__(self):
        self.gru = GRU()

    def set_look_forward(self, value):
        self.gru.look_forward = value
        return self

    def set_n_layers(self, value):
        self.gru.n_layers = value
        return self

    def set_epochs(self, value):
        self.gru.epochs = value
        return self

    def set_batch_size(self, value):
        self.gru.batch_size = value
        return self

    def set_look_backward(self, value):
        self.gru.look_backward = value
        return self

    def get_result(self):
        return self.gru



