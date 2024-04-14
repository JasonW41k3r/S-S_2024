import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad

def Rect(A, tao, T, t):
    temp_t = np.abs(t) % T
    # Vectorized handling using np.where
    return np.where((temp_t <= tao/2) | (temp_t >= T-tao/2), A, 0)

def aNt(N, A, omega, T, tao):
    an = np.zeros(N+1)
    bn = np.zeros(N+1)
    n_gauss = max(50, 10 * N)  # Dynamically set number of Gauss points based on N

    for n in range(N+1):
        FuncA = lambda t: Rect(A, tao, T, t) * np.cos(n * omega * t) * 2 / T
        FuncB = lambda t: Rect(A, tao, T, t) * np.sin(n * omega * t) * 2 / T
        an[n] = fixed_quad(FuncA, -T/2, T/2, n=n_gauss)[0]
        bn[n] = fixed_quad(FuncB, -T/2, T/2, n=n_gauss)[0]

    return an, bn

def plot_xNt(an, bn, omega, T, N, tao):
    tt = np.linspace(-3*T, 3*T, 2000)
    plt.figure(figsize=(15, 6))
    plt.plot(tt, Rect(A, tao, T, tt), label="Original Signal", linestyle='dashed')
    fnt = an[0] / 2
    for n in range(1, N+1):
        fnt += an[n] * np.cos(n * omega * tt) + bn[n] * np.sin(n * omega * tt)
    plt.plot(tt, fnt, label=f"N={N} Fourier Series Approximation")
    plt.title(f"Fourier Series Approximation with N={N} Terms")
    plt.xlabel("Time t")
    plt.ylabel("Signal Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

# Parameters
T = 4
tao = 2
omega = 2 * np.pi / T
A = 1
Ns = [10, 100, 1000]

for N in Ns:
    an, bn = aNt(N, A, omega, T, tao)
    plot_xNt(an, bn, omega, T, N, tao)
