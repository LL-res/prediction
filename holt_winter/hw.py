import time

from sklearn.metrics import mean_squared_error

from data_utils import transformer

class HW:
    def __init__(self):
        self.look_backward = 100
        self.look_forward = 100
        self.alpha = 0.716
        self.beta = 0.029
        self.gamma = 0.993
        self.slen = 12
        self.predict_values = None

    def initial_trend(self, series):
        sum = 0.0
        for i in range(self.slen):
            sum += float(series[i + self.slen] - series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self, series):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(series) / self.slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(sum(series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
        # compute initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self, series):
        result = []
        seasonals = self.initial_seasonal_components(series)
        for i in range(len(series) + self.look_forward):
            if i == 0:  # initial values
                smooth = series[0]
                trend = self.initial_trend(series)
                result.append(series[0])
                continue
            if i >= len(series):  # we are forecasting
                m = i - len(series) + 1
                result.append((smooth + m * trend) + seasonals[i % self.slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (
                            smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                result.append(smooth + trend + seasonals[i % self.slen])
        return result

    def predict(self, series):
        series = series[-self.look_backward:]
        start = time.perf_counter()
        predict = self.triple_exponential_smoothing(series)
        end = time.perf_counter()
        print("time consuming {}ms".format((end - start) * 1000))
        self.predict_values = predict[-self.look_forward:]
        return predict[-self.look_forward:]

    def get_MSE(self,real_values):
        trans_real = transformer.data_transformer(real_values)
        trans_pred = transformer.data_transformer(self.predict_values)
        mse = mean_squared_error(trans_real.normalize(), trans_pred.normalize())
        print("mse : {}".format(mse))
        return mse


class HWBuilder:
    def __init__(self):
        self.hw = HW()

    def set_look_backward(self, value):
        self.hw.look_backward = value
        return self

    def set_look_forward(self, value):
        self.hw.look_forward = value
        return self

    def set_alpha(self, value):
        self.hw.alpha = value
        return self

    def set_beta(self, value):
        self.hw.beta = value
        return self

    def set_gamma(self, value):
        self.hw.gamma = value
        return self

    def set_slen(self, value):
        self.hw.slen = value
        return self

    def get_result(self):
        return self.hw
