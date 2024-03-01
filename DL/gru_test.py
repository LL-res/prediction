import unittest

from matplotlib import pyplot as plt

from ali_data_prepare import ali_data
import gru
from data_utils import generator
from holt_winter import hw


class TestGRU(unittest.TestCase):
    # container_id = 'c_4'
    # data_type = 'cpu'
    # def setUp(self):
    #     self.raw_data = ali_data.get_data(self.container_id, data_type=self.data_type,compress_to_size=200, extend_to_size=2000)
    def test_train(self):
        gru_instance = (gru.GRUBuilder().set_look_backward(120).
         set_look_forward(60).
         set_epochs(30).
         set_batch_size(10).
         set_n_layers(1).get_result())
        gru_instance.prepare_train_data(self.raw_data)
        gru_instance.train(self.container_id+'~'+self.data_type)

    def test_HW_GRU(self):
        gru_instance = (gru.GRUBuilder().set_look_backward(self.look_backward).
                        set_look_forward(self.look_forward).
                        set_epochs(30).
                        set_batch_size(10).
                        set_n_layers(1).get_result())
        hw_look_backward = 400
        hw_instance = (hw.HWBuilder().
                       set_look_backward(hw_look_backward).
                       set_look_forward(self.look_forward).
                       set_slen(200).
                       get_result())
        use_to_predict_GRU = self.raw_data[-(self.look_forward + self.look_backward):-self.look_forward]
        use_to_predict_HW = self.raw_data[-(self.look_forward + hw_look_backward):-self.look_forward]
        use_to_validate = self.raw_data[-self.look_forward:]
        predict_value_GRU = gru_instance.predict(use_to_predict_GRU, 'tt')
        predict_value_HW = hw_instance.predict(use_to_predict_HW)
        plt.figure(dpi=800)
        plt.plot(predict_value_GRU, label='GRU predicted value')
        plt.plot(predict_value_HW, label='Holt-Winter predicted value')
        plt.plot(use_to_validate, label='Actual value')
        plt.xlabel("Timestamp")
        plt.xlabel("Timestamp")
        if self.data_type == 'cpu':
            plt.ylabel("CPU usage(%)")
        else:
            plt.ylabel("Memory usage(%)")
        legend = plt.legend(loc='lower left', prop={'size': 8})

        legend.get_frame().set_alpha(0.7)

        plt.show()

    def test_init(self):
        gru_instance = gru.GRU(json={'noEx':1,'epochs':2})
        print(gru_instance)

    def setUp(self):
        self.data_type = 'cpu'
        self.look_forward = 60
        self.look_backward = 200
        self.raw_data = ali_data.get_data('c_25', data_type=self.data_type,compress_to_size=200, extend_to_size=2000+self.look_forward,draw_pic=True)
        self.raw_data = generator.generate(self.raw_data,1)

