import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

n1 = np.linspace(0, 8, 9)
x1 = [3, 2, 1, -2, 1, 0, 4, 0, 3]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plt.subplot(221)
plt.stem(n1,x1,'-',use_line_collection=True)
plt.grid(True)
plt.title('x[n]')

n2 = np.linspace(0, 6, 7)
x2 = [1, -2, 3, -4, 3, 2, 1]
plt.subplot(222)
plt.stem(n2,x2,'-',use_line_collection=True)
plt.grid(True)
plt.xticks(np.arange(0, 7, step=1.0))
plt.title('h[n]')

plt.subplot(212)
y = sg.convolve(x1, x2,'full')
n3 = np.linspace(0, 14, 15)
plt.stem(n3,y,'-',use_line_collection=True)
plt.grid(True)
plt.title('Conv Sum y[n]')

plt.xlabel('Time index n')
plt.subplots_adjust(top=1, wspace=0.2, hspace=0.2)
plt.show()