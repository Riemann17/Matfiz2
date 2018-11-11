import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# parameters
n = 20
a = 0
b = 1

# constants
k1 = 2
k2 = 1
k3 = 2

m1 = 2
m2 = 3
m3 = 2

q1 = 3
q2 = 2
q3 = 1

p1 = 1
p2 = 2
p3 = 3

alph1 = 1
alph2 = 1

mu1 = m1*m2*(-k3 - k1*np.cos(k2*a))*np.cos(m2*a) + \
        alph1*(m3 + m1*np.sin(m2*a))
mu2 = m1*m2*(k3 + k1*np.cos(k2*b))*np.cos(m2*b) + \
      + alph2*(m3 + m1*np.sin(m2*b))


def k(x):
    return k1*np.cos(k2*x) + k3


def p(x):
    return p1*np.sin(p2*x) + p3


def q(x):
    return q1*np.cos(q2*x) + q3


def f(x):
    return k1*k2*m1*m2*np.cos(m2*x)*np.sin(k2*x) + \
            m1*m2**2*(k3 + k1*np.cos(k2*x))*np.sin(m2*x) + \
            (q3 + q1*np.cos(q2*x))*(m3 + m1*np.sin(m2*x)) + \
            m1*m2*np.cos(m2*x)*(p3 + p1*np.sin(p2*x))


def u(x):
    return m1*np.sin(m2*x) + m3


x = np.linspace(a, b, num=n+1)
h = (b - a) / n


def integrate(f, x1, x2):
    return (f(x1) + f(x2))*(x2 - x1) / 2


def ai(i):
    return (integrate(k, x[i-1], x[i]) - integrate(lambda t: q(t)*(x[i] - t)*(t - x[i-1]), x[i-1], x[i]))/h


def di(i):
    return (integrate(lambda t: q(t)*(t - x[i-1]), x[i-1], x[i]) + integrate(lambda t: q(t)*(x[i+1] - t), x[i], x[i+1]))/(h*h)


def cp_i(i):
    return integrate(lambda t: p(t)*(x[i+1] - t), x[i], x[i+1])/(h*h)


def cm_i(i):
    return integrate(lambda t: p(t) * (t - x[i-1]), x[i-1], x[i]) / (h * h)


def phii(i):
    return (integrate(lambda t: f(t)*(t - x[i-1]), x[i-1], x[i]) + integrate(lambda t: f(t)*(x[i+1] - t), x[i], x[i+1]))/(h*h)


def plot2(f, xi, y, a, b, n=50):
    u = [f(x) for x in xi]

    plt.figure(figsize=(12, 6), num='pic')
    plt.grid()
    plt.plot(xi, u, 'r', linewidth=2, label=r'$\ u(x)$')
    plt.plot(xi, y, '--b', linewidth=1.5, label=r'$\ y(x)$')

    plt.legend(fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def demo():
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)

    A[0][0] = alph1*h + integrate(lambda t: q(t)*(x[1] - t), x[0], x[1]) + ai(1) - cp_i(0)*h
    A[0][1] = -(ai(1) - cp_i(0)*h)
    b[0] = integrate(lambda t: f(t)*(x[1] - t), x[0], x[1]) + mu1*h

    for i in range(1, n):
        A[i][i-1] = ai(i) / (h*h) + cm_i(i)/h
        A[i][i] = - ((ai(i+1) + ai(i))/(h*h) + di(i) - (cp_i(i) - cm_i(i))/h)
        A[i][i+1] = ai(i+1) / (h*h) - cp_i(i)/h
        b[i] = -phii(i)

    A[n][n-1] = -(ai(n) + cm_i(n)*h)
    A[n][n] = ai(n) + cm_i(n)*h + alph2*h + integrate(lambda t: q(t)*(t - x[n-1]), x[n-1], x[n])
    b[n] = mu2*h + integrate(lambda t: f(t)*(t - x[n-1]), x[n-1], x[n])

    y = solve(A, b)

    print("x:       u:         y:         delta:")
    for i in range(n+1):
        xi = x[i]
        yi = y[i]
        ui = u(xi)
        delta = abs(ui - yi)
        print("{0:.3f}    {1:.5f}    {2:.5f}    {3:.5f}".format(xi, ui, yi, delta))

    plot2(u, x, y, a, b)


if __name__ == "__main__":
    demo()

