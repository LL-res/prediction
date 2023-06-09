import math

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

import param

class GRUParameters:
    def __init__(self, d):
       self.__dict__ = d

#gru_params = json.loads(sys.stdin.read(), object_hook=GRUParameters)

metrics = []
for i in range(1,10000):
    metrics.append(math.sin(0.01*i))
tests = []
for i in range(10000,12000):
    tests.append(math.cos(0.01*i))

def train_data_prepare(metrics):
    """
        param : metrcis list
    """
    #如修改标签维数，需再进行一次滑动窗口
    metrics = np.array(metrics)

    #去掉标签数据用不到的部分
    labels_t = metrics[param.look_back:]
    #因为标签只有一个数，所以去掉最后一个输入用不上的数
    inputs_t = metrics[:-1]

    inputs = np.array([])
    #滑动窗口，窗口大小为look_back
    for i in range(len(inputs_t)-param.look_back+1):
        element = metrics[i:i+param.look_back]
        #这个append会把元素全部展开进行append
        inputs = np.append(inputs,element)
    #[[[1.],  [2.],  [3.]],, [[2.],  [3.],  [4.]],, [[3.],  [4.],  [5.]],, [[4.],  [5.],  [6.]]]
    inputs = inputs.reshape((-1,param.look_back,1))
    #[[4], [5], [6], [7]]
    labels = labels_t.reshape((-1,1))

    train_data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=param.batch_size, drop_last=True)

    return train_loader

# train_loader = train_data_prepare(metrics)
# for x,y in train_loader:
#     print('x=',x,'y=',y)
