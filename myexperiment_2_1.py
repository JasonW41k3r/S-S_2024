import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

fs = 1000
t1 = np.array([t/fs for t in range(-1100,2101)])
x1 = np.array([t if t>=0 else 0 for t in t1])
t2 = np.array([t/fs for t in range(-1100,2101)])
x2 = np.array([np.exp(-1 * t) if t>=0 else 0 for t in t2])
y1 = sg.convolve(x1,x2)/fs # 卷积
n = len(y1) # 卷积结果采样点数量
tt = np.linspace(-2200,4201,n)/fs # 定义新序列时间范围，卷积结果时间轴，卷积左端点=x1左端点+x2左端点，卷积右端点=x1右端点+x2右端点-1，
fig, axs = plt.subplots(2, 2, figsize=(10, 10)) # 通过figsize调整图大小
plt.subplots_adjust(wspace = 0.2, hspace = 0.2) # 通过wspace和hspace调整子图间距
plt.subplot(221) # 绘制x1(t)信号的子图
plt.plot(t1,x1) # 绘制x1(t)信号
plt.grid() # 显示网格
_ = plt.title('x1(t)') # x1(t)信号title
plt.subplot(222) # 绘制x2(t)信号的子图
plt.plot(t2,x2) # 绘制x2(t)信号
plt.grid() # 显示网格
_ = plt.title('x2(t)') # x2(t)信号title
plt.subplot(212) # 绘制卷积信号的子图
plt.plot(tt,y1) # 绘制卷积信号
plt.grid() # 显示网格
_ = plt.title('conv(x1,x2)') # 卷积信号title
plt.show() # 显示图像