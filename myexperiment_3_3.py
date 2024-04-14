import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.fft import fft, ifft

# 定义三角波生成函数
def triangular_wave(A, T, t):
    temp_t = np.abs(t) % T
    return np.where(temp_t <= T/2, (4*A/T)*temp_t - A, (-4*A/T)*temp_t + 3*A)

# 定义系统的频率响应
def frequency_response(omega):
    return 1 / (1j * omega + 2)

# 时间轴参数
T = 10  # 三角波周期
A = 1   # 三角波振幅
t = np.linspace(-T, T, 1000)  # 时间数组

# 生成三角波
x_t = triangular_wave(A, T, t)

# 计算频率响应 H(jω) 在不同频率ω上的值
omega = np.linspace(-20, 20, 1000)  # 频率范围
H_jw = frequency_response(omega)

# 计算x(t)的FFT
X_f = fft(x_t)

# 使用频率响应 H(jω) 计算 Y(f)
Y_f = X_f * frequency_response(2 * np.pi * np.fft.fftfreq(len(t), d=t[1]-t[0]))

# 逆FFT得到时域中的输出 y(t)
y_t = ifft(Y_f)

# 绘制输入x(t)和输出y(t)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(t, x_t, label='Input x(t)')
plt.title('Input Triangular Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, y_t.real, label='Output y(t)')  # 只绘制实部
plt.title('Output Response')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
