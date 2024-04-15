import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 定义新的矩形波函数
def Rect(A, tao, T, t):
    temp_t = np.abs(t) % T
    return np.where((temp_t <= tao/2) | (temp_t >= T-tao/2), A, 0)

# 参数设定
T = 4  # 基波周期
tao = 2  # 有效宽度
omega = 2 * np.pi / T  # 基波频率
A = 1  # 幅度
N = 10  # 谐波数

# 计算傅里叶级数系数
Xn = np.zeros(2*N+1)  # 傅里叶级数指数形式的幅值
nn = np.arange(-N, N+1)  # 傅里叶级数项数-N~N

# 积分函数
def integrand(t, n, omega, A, T, tao):
    return Rect(A, tao, T, t) * np.exp(-1j * n * omega * t)

# 对每个n计算傅里叶系数
for n in range(-N, N+1):
    func = lambda t: integrand(t, n, omega, A, T, tao)
    # 积分区间为一个完整周期
    Xn[n + N] = quad(func, 0, T)[0] / T

# 绘制频谱图
plt.figure(figsize=(10, 5))
plt.stem(nn, np.abs(Xn), use_line_collection=True)  # 绘制幅值频谱图
plt.title("周期三角信号的傅里叶级数幅度频谱")
plt.xlabel("谐波数 n")
plt.ylabel("幅值 |Xn|")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.grid(True)
plt.show()
