import numpy as np
class data_transformer():
    def __init__(self,data_list):
        self.data_list = data_list
        self.np_data = np.array(data_list)
    def normalize(self):
        res_np = (self.np_data - self.np_data.min()) / (self.np_data.max() - self.np_data.min())
        return res_np.tolist()
    def denormalize(self,predict_list):
        predict_np = np.array(predict_list)
        res_np = predict_np * (self.np_data.max() - self.np_data.min()) + self.np_data.min()
        return res_np.tolist()