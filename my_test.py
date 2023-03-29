import data_preparation
import net
import math
import matplotlib.pyplot as plt

metrics = []
for i in range(0,1000):
    metrics.append(100*math.sin(0.01*i))
tests = []
for i in range(0,100):
    tests.append(100*math.sin(0.01*i))
real = []
for i in range(0,200):
    real.append(100*math.sin(0.01*i))
train_loader = data_preparation.train_data_prepare(metrics)

#net.train(train_loader,"cpu")

for i in range(100,200):
    out = net.predict(tests,"GRU","cpu")
    tests.append(out.item())


plt.plot(tests)
plt.plot(real)
plt.show()