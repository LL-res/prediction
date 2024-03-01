import unittest

from DL import gru
from holt_winter import hw
from test.evaluator import evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.look_forward = 60
        self.gru_instance = (gru.GRUBuilder().set_look_backward(400).
                        set_look_forward(self.look_forward).
                        set_epochs(30).
                        set_batch_size(10).
                        set_n_layers(1).get_result())
        self.hw_instance = (hw.HWBuilder().
                       set_look_backward(400).
                       set_look_forward(self.look_forward).
                       set_slen(200).
                       get_result())

    def test_evaluate(self):
        to_evaluate = [('c_12','cpu'),('c_12','mem'),('c_14','mem'),('c_25','cpu'),('c_25','mem'),('c_4','cpu'),('c_4','mem')]
        for item in to_evaluate:
            evaluator_instance = evaluator(self.hw_instance,self.gru_instance,item[0],item[1])
            evaluator_instance.evaluate()
            evaluator_instance.draw()