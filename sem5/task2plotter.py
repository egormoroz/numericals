from task2poly import *
from task2spline import *
import matplotlib.pyplot as plt

a, b = 1, 2
n, M = 10, 1000
f = lambda x: x + np.log10(x/5)

x_test = np.linspace(a, b, M)
y_test = f(x_test)

x_lp = np.linspace(a, b, n)
x_opt = alterspace(a, b, n)

y_polylp = ipoly_lagr(x_lp, f(x_lp), x_test)
y_polyopt = ipoly_lagr(x_opt, f(x_opt), x_test)

_, axs = plt.subplots(1, 2)

k = -1
axs[0].plot(x_test, y_test, 'r', label='f(x)')
axs[0].plot(x_test, y_polylp, 'b--', label='ipoly')
axs[0].plot(x_test, y_polyopt, 'g-.', label='ipoly_polyopt')
axs[0].legend(loc='lower right')

axs[1].plot(x_test, y_polylp - y_test, 
     'b', label='P(x) - f(x)')
axs[1].plot(x_test, y_polyopt - y_test, 
     'g', label='Popt(x) - f(x)')
axs[1].plot(x_test, np.zeros_like(x_test), '--')
axs[1].legend(loc='lower right')

plt.show()


y_splp = spl32(x_lp, f(x_lp), x_test)
y_spopt = spl32(x_opt, f(x_opt), x_test)

plt.plot(x_test, y_polylp - y_test, 
     'r', label='P(x) - f(x)')
plt.plot(x_test, y_polyopt - y_test, 
     'g', label='Popt(x) - f(x)')
plt.plot(x_test, y_splp - y_test, 
     'b', label='S32(x) - f(x)')
plt.plot(x_test, y_spopt - y_test, 
     'm', label='S32opt(x) - f(x)')
plt.plot(x_test, np.zeros_like(x_test), '--')
plt.legend(loc='lower right')

plt.show()

