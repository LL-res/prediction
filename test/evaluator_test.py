import unittest

from matplotlib import pyplot as plt, font_manager

from DL import gru
from holt_winter import hw
from test.evaluator import evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.look_forward = 60
        self.gru_instance = (gru.GRUBuilder().set_look_backward(400).
                             set_look_forward(self.look_forward).
                             set_epochs(20).
                             set_batch_size(10).
                             set_n_layers(2).get_result())
        self.hw_instance = (hw.HWBuilder().
                            set_look_backward(400).
                            set_look_forward(self.look_forward).
                            set_alpha(0.716).
                            set_beta(0.21).
                            set_gamma(0.32).
                            set_slen(200).
                            get_result())

    def test_evaluate(self):
        to_evaluate = [('c_12','cpu'),('c_25','cpu'),('c_4','mem')]
        to_evaluate = [('c_14','mem'),('c_25','mem'),('c_4','cpu')]
        to_evaluate = [('c_25','mem')]
        for item in to_evaluate:
            evaluator_instance = evaluator(self.hw_instance, self.gru_instance, item[0], item[1])
            evaluator_instance.evaluate()
            evaluator_instance.draw()

    def test_chinese_draw(self):
        plt.figure(dpi=1000)
        plt.rcParams['font.sans-serif'] = ['SimSun']
        to_plot = [x for x in range(30)]
        plt.xlabel('这是中文，english label')
        plt.plot(to_plot)
        plt.show()

    import matplotlib.pyplot as plt

    def test_plot_two(self):
        plt.figure(dpi=300, figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False

        gru_mse = [0.87, 0.93, 1.49, 1.91, 0.92, 1.11]
        hw_mse = [1.93, 6.88, 3.46, 2.74, 3.69, 5.15]

        x_labels = ['a', 'b', 'c', 'd', 'e', 'f']
        x = range(len(x_labels))

        # 设置图表标题和轴标签
        #plt.title('GRU vs Holt-Winter MSE Comparison', fontsize=16)
        plt.xlabel('容器编号', fontsize=14)
        plt.ylabel('MSE', fontsize=14)

        # 绘制折线图并突出显示线上的点
        plt.plot(x, gru_mse, label='GRU', marker='o', markersize=8, linestyle='-', linewidth=2, color='blue')
        plt.plot(x, hw_mse, label='Holt-Winter', marker='s', markersize=8, linestyle='--', linewidth=2, color='red')

        # 设置横坐标标签
        plt.xticks(x, x_labels, fontsize=12)
        plt.yticks(fontsize=12)

        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加图例
        plt.legend(fontsize=12)

        # 调整布局
        plt.tight_layout()

        plt.show()


    def test_plot_relative(self):
        relative_improvement = []
        gru_mse = [0.00563,0.00898,0.00918,0.00412,0.00858,0.04262]
        hw_mse = [0.00735,0.01205,0.00934,0.00614,0.01103,0.04266]
        # 计算每对MSE值的指标
        for gru, hw in zip(gru_mse, hw_mse):
            # 相对改善
            if hw != 0:
                rel_imp = (hw - gru) / hw * 100
                relative_improvement.append(rel_imp)
    def test_clear(self):
        evaluator.clear_results()
