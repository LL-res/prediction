import unittest
import hw
from ali_data_prepare import ali_data
import matplotlib.pyplot as plt
from data_utils import generator



class TestHW(unittest.TestCase):

    def test_predict_ali(self):
        period,look_forward,look_backward = 24,24,100
        raw_data = ali_data.get_data('c_4',compress_to_size=period,extend_to_size=200)

        hw_instance = (hw.HWBuilder().
                       set_look_backward(look_backward).
                       set_look_forward(look_forward).
                       set_slen(period).
                       get_result())

        predict_values = hw_instance.predict(raw_data[:-look_forward])
        real_values = raw_data[-look_forward:]
        hw_instance.get_MSE(real_values)

        plt.figure(dpi=800)
        plt.plot(predict_values,label='Predicted Values')
        plt.plot(real_values,label='Actual Values')
        plt.xlabel("Time Point(3s/Interval)")
        plt.ylabel("RPS")
        plt.legend()
        plt.show()

    def test_predict_generator(self):
        period, look_forward, look_backward = 12, 60, 120
        raw_data = generator.generate([],0,total_size=200)
        hw_instance = (hw.HWBuilder().
                       set_look_backward(look_backward).
                       set_look_forward(look_forward).
                       set_slen(period).
                       get_result())

        predict_values = hw_instance.predict(raw_data[:-look_forward])
        real_values = raw_data[-look_forward:]
        hw_instance.get_MSE(real_values)

        plt.figure(dpi=800)
        plt.plot(predict_values, label='Predicted Values')
        plt.plot(real_values, label='Actual Values')
        plt.xlabel("Time Point(3s/Interval)")
        plt.ylabel("RPS")
        plt.legend()
        plt.show()