import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad

def Rect(A, Tao, T, t):
    # 使用向量化操作处理输入数组t
    temp_t = np.abs(t) % T
    return np.where(temp_t <= T/2, (4*A/T)*temp_t - A, (-4*A/T)*temp_t + 3*A)

def aNt(N, A, omega, T, tao):
    an = np.zeros(N+1)
    bn = np.zeros(N+1)
    # 根据N值动态设置高斯积分点数
    n_gauss = max(50, N * 5)  # 基本点数设置为N的5倍，至少为50

    FuncA0 = lambda t : Rect(A, tao, T, t) * 2 / T
    an[0] = fixed_quad(FuncA0, -T/2, T/2, n=n_gauss)[0]

    for n in range(1, N+1):
        FuncA = lambda t: Rect(A, tao, T, t) * np.cos(n * omega * t) * 2 / T
        FuncB = lambda t: Rect(A, tao, T, t) * np.sin(n * omega * t) * 2 / T
        an[n] = fixed_quad(FuncA, -T/2, T/2, n=n_gauss)[0]
        bn[n] = fixed_quad(FuncB, -T/2, T/2, n=n_gauss)[0]

    return an, bn

def plot_xNt(an, bn, omega, T, N, tao):
    tt = np.arange(-3*T, 3*T, 0.01)
    plt.figure(figsize=(10, 6))
    original_signal = np.array([Rect(A, tao, T, t) for t in tt])
    plt.plot(tt, original_signal, label="Original Signal", linestyle='dashed')
    fnt = an[0] / 2
    for n in range(1, N + 1):
        fnt += an[n] * np.cos(n * omega * tt) + bn[n] * np.sin(n * omega * tt)
    plt.plot(tt, fnt, label=f"Approximation N={N}")
    plt.title(f"Approximation with N={N} terms")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(True)
    plt.show()

T = 4
tao = 2
omega = 2 * np.pi / T
A = 2
N_list = [10, 100, 1000]

for N in N_list:
    an, bn = aNt(N, A, omega, T, tao)
    plot_xNt(an, bn, omega, T, N, tao)
