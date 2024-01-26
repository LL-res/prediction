import unittest
from ali_data_prepare import ali_data
import gru
class TestGRU(unittest.TestCase):
    def setUp(self):
        self.raw_data = ali_data.get_data('c_4', compress_to_size=200, extend_to_size=2000)
    def test_train(self):
        gru_instance = (gru.GRUBuilder().set_look_backward(120).
         set_look_forward(60).
         set_epochs(30).
         set_batch_size(10).
         set_n_layers(1).get_result())
        gru_instance.prepare_train_data(self.raw_data)
        gru_instance.train('cpu')

    def test_init(self):
        gru_instance = gru.GRU(json={'noEx':1,'epochs':2})
        print(gru_instance)

