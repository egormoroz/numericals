import numpy as np
import matplotlib.pyplot as plt

def parse_array(s):
    return np.array(list(map(float, s.strip().split())), 
            dtype=np.float64)


with open('lagr.txt', 'r') as f:
    data = []
    x_test = parse_array(f.readline())
    y_test = parse_array(f.readline())
    for i in f:
        data.append(parse_array(i))
    ip_test, ip_test_opt = np.array(data[::2]), np.array(data[1::2])

_, axs = plt.subplots(1, 2)

k = -1
axs[0].plot(x_test, y_test, 'r', label='f(x)')
axs[0].plot(x_test, ip_test[k], 'b--', label='ipoly')
axs[0].plot(x_test, ip_test_opt[k], 'g-.', label='ipoly_opt')
axs[0].legend(loc='lower right')

axs[1].plot(x_test, ip_test[k] - y_test, 
     'b', label='P(x) - f(x)')
axs[1].plot(x_test, ip_test_opt[k] - y_test, 
     'g', label='Popt(x) - f(x)')
axs[1].plot(x_test, np.zeros_like(x_test), '--')
axs[1].legend(loc='lower right')

plt.show()

