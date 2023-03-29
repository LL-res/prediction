import data_preparation
import net
import math
import matplotlib.pyplot as plt

import param

metrics = []

for i in range(0,10000):
    metrics.append(100*math.sin(0.1*i))
tests = []
for i in range(0,param.look_back):
    tests.append(100*math.sin(0.1*i))
real = []
for i in range(0,param.look_back+param.look_forward):
    real.append(100*math.sin(0.1*i))
train_loader = data_preparation.train_data_prepare(metrics)

#net.train(train_loader,"cpu")


out = net.predict(tests,"GRU","cpu")
print(out)
tests.append(out.item())


plt.plot(tests)

plt.plot(real)
plt.show()