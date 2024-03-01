import csv
import os.path


class result:
    def __init__(self,model_type):
        # hw 或 gru
        self.model_type = model_type
        self.mse = None
        #单位为ms
        self.train_time_consumed = None
        self.predict_time_consumed = None
        #cpu 或 mem
        self.data_type = None
        self.container_id = None

    def save(self):
        results_path = './results.csv'
        #检查results是否存在
        if os.path.exists(results_path) is False:
            results_header = ['model_type', 'container_id','data_type','mse', 'train_time_consumed(s)','predict_time_consumed(ms)']
            with open(results_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(results_header)
        with open(results_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            data_to_write = [self.model_type,self.container_id,self.data_type,self.mse,self.train_time_consumed,self.predict_time_consumed]
            csv_writer.writerow(data_to_write)

class resultBuilder:
    def __init__(self,model_type):
        self.result = result(model_type)

    def set_mse(self,mse):
        self.result.mse = mse
        return self

    def set_train_time_consumed(self,time_consumed):
        self.result.train_time_consumed = time_consumed
        return self

    def set_predict_time_consumed(self,time_consumed):
        self.result.predict_time_consumed = time_consumed
        return self

    def set_data_type(self,data_type):
        self.result.data_type = data_type
        return self

    def set_container_id(self,container_id):
        self.result.container_id= container_id
        return self

    def get_result(self):
        return self.result


