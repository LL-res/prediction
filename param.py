import torch

look_back = 100
look_forward = 100
batch_size = 1000
n_layers = 2
epochs = 10

STATUS_TRAIN =  1
STATUS_PREDICT = 2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")