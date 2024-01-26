import ali_data
import numpy as np
import matplotlib.pyplot as plt

import unittest


class TestAliData(unittest.TestCase):

    def test_get_data(self):
        data = ali_data.get_data('c_4',72,200)
        plt.plot(data)
        #plt.scatter(np.arange(len(data)), data, label='Original Data', color='blue', alpha=0.5)
        plt.show()




