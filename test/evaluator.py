import csv
import os.path
import shutil
import time

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from DL import gru
from ali_data_prepare import ali_data
from data_utils import transformer
from test.result import resultBuilder
from data_utils import generator


class evaluator:
    container_id = None
    data_type = None
    hw_prediction = []
    gru_prediction = []
    real_data = []
    use_data = []
    hw_instance = None
    gru_instance = None

    def __init__(self, hw_instance, gru_instance, container_id, data_type):
        self.container_id = container_id
        self.data_type = data_type
        self.raw_data = ali_data.get_data(self.container_id, data_type=self.data_type, compress_to_size=200,
                                          extend_to_size=2000 + gru_instance.look_forward, draw_pic=False)
        self.raw_data = generator.generate(self.raw_data, 1, upbound=100)
        self.hw_instance = hw_instance
        self.gru_instance = gru_instance
        self.real_data = self.raw_data[-gru_instance.look_forward:]
        self.use_data = self.raw_data[:-gru_instance.look_forward]

    def get_MSE(self, real, prediction):
        # trans_real = transformer.data_transformer(real)
        # trans_pred = transformer.data_transformer(prediction)
        # mse = mean_squared_error(trans_real.normalize(), trans_pred.normalize())
        mse = mean_squared_error(real, prediction)
        return mse

    def evaluate(self):
        pt_name = self.container_id + '~' + self.data_type
        gru_train_duration = None
        if os.path.exists('./' + pt_name + '.pt') is False:
            self.gru_instance.prepare_train_data(self.use_data)
            start = time.time()
            self.gru_instance.train(pt_name)
            end = time.time()
            gru_train_duration = end - start
        start = time.time()
        self.gru_prediction = self.gru_instance.predict(self.use_data, pt_name)
        end = time.time()
        gru_prediction_duration = 1000 * (end - start)
        gru_mse = self.get_MSE(self.real_data, self.gru_prediction)
        gru_result = (resultBuilder('GRU').set_mse(gru_mse).
                      set_container_id(self.container_id).
                      set_data_type(self.data_type).
                      set_train_time_consumed(gru_train_duration).
                      set_predict_time_consumed(gru_prediction_duration).
                      set_real_value(self.real_data).
                      set_predict_value(self.gru_prediction).
                      get_result())
        gru_result.save()

        start = time.time()
        self.hw_prediction = self.hw_instance.predict(self.use_data)
        end = time.time()
        hw_prediction_duration = 1000 * (end - start)
        hw_mse = self.get_MSE(self.real_data, self.hw_prediction)
        hw_result = (resultBuilder('HW').set_mse(hw_mse).
                     set_container_id(self.container_id).
                     set_data_type(self.data_type).
                     set_predict_time_consumed(hw_prediction_duration).
                     set_real_value(self.real_data).
                     set_predict_value(self.hw_prediction).
                     get_result())
        hw_result.save()

    def draw(self):
        plt.rcParams['font.sans-serif'] = ['simsun']
        plt.figure(dpi=1000)
        plt.plot(self.gru_prediction, label='GRU 预测值')
        plt.plot(self.hw_prediction, label='Holt-Winter 预测值')
        plt.plot(self.real_data, label='真实值')
        plt.xlabel("时间戳(min)")
        if self.data_type == 'cpu':
            plt.ylabel('CPU使用率(%)')
        else:
            plt.ylabel('内存使用率(%)')
        # legend = plt.legend(loc='lower left', prop={'size': 8})
        legend = plt.legend(prop={'size': 8})
        legend.get_frame().set_alpha(0.7)
        plt.show()

    @staticmethod
    def clear_results():
        results_path = 'results.csv'
        if os.path.exists(results_path) is False:
            print('结果集未创建')
            return
        shutil.copy(results_path, "resultTemp.csv")
        # 删除resultTemp.csv中所有第一列为“mark”的行
        temp_file = "resultTemp.csv"
        rows_to_keep = []
        with open(temp_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[4] is not None:
                    rows_to_keep.append(row)
                    continue
                if row[0] != "HW" and row[0] != "GRU":
                    rows_to_keep.append(row)

        with open(temp_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows_to_keep)

        # 删除result.csv
        os.remove("results.csv")

        # 重命名resultTemp.csv为result.csv
        os.rename("resultTemp.csv", "results.csv")

        print("操作完成")
