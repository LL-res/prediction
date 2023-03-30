import data_preparation
import net
import math
import matplotlib.pyplot as plt

import param

metrics = []
start_predict_idex = 4000

for i in range(0,1000):
    metrics.append(100*math.sin(0.1*i))
tests = []
for i in range(start_predict_idex,start_predict_idex+param.look_back):
    tests.append(100*math.sin(0.1*i))
real = []
for i in range(start_predict_idex,start_predict_idex+param.look_back+param.look_forward):
    real.append(100*math.sin(0.1*i))
#train_loader = data_preparation.train_data_prepare(metrics)

#net.train(train_loader,"cpu")


out = net.predict(tests,"GRU","cpu")
print(out.tolist())
tests.extend(out.tolist()[0])

plt.plot(tests)

plt.plot(real)
plt.show()