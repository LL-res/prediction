import torch

look_back = 10
look_forward = 10
batch_size = 10
n_layers = 2

STATUS_TRAIN =  1
STATUS_PREDICT = 2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")