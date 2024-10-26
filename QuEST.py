import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

# 設置樣本數據
n = 10  # 假設樣本數
p = 5   # 假設維度
t = np.random.uniform(0, 1, p)  # 樣本特徵值的範例數據

def stieltjes_transform(z, t, n, p):
    """
    計算公式 (2.14) 的 Stieltjes 變換
    :param z: 複數變量
    :param t: 樣本特徵值列表
    :param n: 樣本數
    :param p: 維度
    :return: Stieltjes 變換 m
    """
    def func(m):
        return m - (1 / p) * np.sum(t / (t * (1 - p / n * m) - z))
    
    # 使用求解器找到唯一解
    solution = root(func, 0.5)  # 初始猜測 0.5
    return solution.x[0] if solution.success else None

def f_n_p_x(x, t, n, p):
    """
    計算公式 (2.15) 的分布函數 F_{n,p}^t(x)
    :param x: 實數變量 x
    :param t: 樣本特徵值列表
    :param n: 樣本數
    :param p: 維度
    :return: F_{n,p}^t(x)
    """
    if x == 0:
        term1 = 1 - n / p
        term2 = 1 - (1 / p) * np.sum([1 if ti == 0 else 0 for ti in t])
        return max(term1, term2)
    else:
        def integrand(xi):
            m_t = stieltjes_transform(xi + 1e-10j, t, n, p)
            return np.imag(m_t)
        
        integral, _ = quad(integrand, -np.inf, x)
        return (1 / np.pi) * integral

def inverse_f_n_p(u, t, n, p):
    """
    計算公式 (2.16) 的逆分布函數 (F_{n,p}^t)^{-1}(u)
    :param u: 累積概率
    :param t: 樣本特徵值列表
    :param n: 樣本數
    :param p: 維度
    :return: x 使得 F_{n,p}^t(x) <= u
    """
    x_values = np.linspace(-3, 3, 1000)
    f_values = [f_n_p_x(x, t, n, p) for x in x_values]
    eligible_x = [x for x, F in zip(x_values, f_values) if F <= u]
    return max(eligible_x) if eligible_x else None

def quantized_eigenvalue(i, t, n, p):
    """
    計算公式 (2.17) 的量化特徵值 q_{n,p}^i(t)
    :param i: 第 i 個特徵值
    :param t: 樣本特徵值列表
    :param n: 樣本數
    :param p: 維度
    :return: q_{n,p}^i(t)
    """
    lower_bound = (i - 1) / p
    upper_bound = i / p
    def integrand(u):
        return inverse_f_n_p(u, t, n, p)
    
    integral, _ = quad(integrand, lower_bound, upper_bound)
    return p * integral

# 測試
z = 1 + 1j  # 複數 z 的例子
m = stieltjes_transform(z, t, n, p)
print("Stieltjes 變換 m:", m)

x = 0.5  # x 的例子
f_val = f_n_p_x(x, t, n, p)
print("分布函數 F_{n,p}^t(x):", f_val)

u = 0.5  # 逆分布函數的例子
inverse_f_val = inverse_f_n_p(u, t, n, p)
print("逆分布函數 (F_{n,p}^t)^{-1}(u):", inverse_f_val)

i = 1  # 第 i 個特徵值的量化
quantized_val = quantized_eigenvalue(i, t, n, p)
print(f"量化特徵值 q_{{n,p}}^{i}(t):", quantized_val)
