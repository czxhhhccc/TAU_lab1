import control
import control.matlab as matlab
import numpy as np
import matplotlib.pyplot as plt
import math
import colorama as color
import pandas as pd
from sympy import *
from numpy.linalg import *
from math import *
from scipy.optimize import fsolve

# 8.1
def p1():
    timeLine = []
    for i in range(0,50000):
        timeLine.append(i/100)
        pass
    y = []
    y1 = []
    for i in timeLine:
        y.append(0.057*exp(-3.18*i)-24.194*cos(0.16*i)+1.293*sin(0.16*i)+24.2609)
        # y1.append(10 * (log10((0.48 * i) ** 2 + 24 ** 2) - log10((1 - 39.2 * i ** 2) ** 2 + (12.35 * i - 12.25 * i ** 3) ** 2)))
        pass
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(timeLine,y)
    # plt.plot(timeLine,y1)
    plt.title('Переходная характеристика \n'
              'при предельном коэффициенте обратной связи')
    ax.grid()
    yticks = np.linspace(0,60,7)
    plt.yticks(yticks)
    xticks = np.linspace(0,700,8)
    plt.xticks(xticks)
    y_max_index = y.index(max(y)) #get the index of the maximum inflation
    y_max = y[y_max_index] # get the inflation corresponding to this index
    time_max = timeLine[y_max_index] # get the year corresponding to this index
    # print(y_max_index)
    text= 'x={:.3f}, y={:.3f}'.format(time_max, y_max)
    ax.annotate(text, xy=(time_max, y_max), xytext=(time_max, y_max+5),
                arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.show()
# p1()
# 8.2

# 8.3
def p3():
    num = [0,-12.04,-0.2407]
    den = [1,12.25,39.2,12.25]
    p = symbols('p')
    w = symbols('w',real = True)
    f1 = poly(sum(coef*p**i for i,coef in enumerate(num)))/poly(sum(coef*p**i for i,coef in enumerate(den)))
    f2 = f1.subs(p,complex(0, 1)*w)
    print(f2)
    # print(f2)
    f2_real = simplify(re(f2))
    f2_imag = simplify(im(f2))
    print('f2_real',f2_real)
    print('f2_imag',f2_imag)
    w_ = []
    for i in range(0, 3000):
        w_.append(i / 1000)
        pass
    real_ = []
    imag_ = []
    # for i in w_:
    #     real_.append(f2_real.subs(w,i))
    #     imag_.append(f2_imag.subs(w, i))
    #     pass
    for i in w_:
        real_.append(i**2*(138.05456*i**2 - 147.2493)/(150.0625*i**6 + 1236.515*i**4 + 71.6625*i**2 + 1.0))
        imag_.append(i*(2.948575*i**4 + 469.019425*i**2 - 12.04)/(150.0625*i**6 + 1236.515*i**4 + 71.6625*i**2 + 1.0))
        pass
    print('max',min(real_))
    plt.plot(real_,imag_)
    plt.title('Подограв Найквиста')
    plt.xlabel('Re')
    plt.ylabel('Im')
    x0 = [-1]
    y0 = [0]
    plt.plot(x0, y0, "s")
    plt.grid()
    plt.show()
# p3()
# i=0.17
# print(i**2*(138.05456*i**2 - 147.2493)/(150.0625*i**6 + 1236.515*i**4 + 71.6625*i**2 + 1.0))
def p45():
    w_1 = []
    for i in range(1, 1000):
        w_1.append(i / 1000)
        pass
    w_2 = []
    for i in range(1200, 100000):
        if i%5 ==0:
            w_2.append(i / 1000)
        pass
    w_ = w_1+w_2
    L = []
    phase = []
    for i in w_1:
        L.append(10*np.log10(((0.2407*i**2)**2+(12.04*i)**2)/((1-39.2*i**2)**2+(-12.25*i**3+12.25*i)**2)))
        phase.append(atan((i*(2.948575*i**4 + 469.019425*i**2 - 12.04))/(i**2*(138.05456*i**2 - 147.2493)))/np.pi*180-180)
    for i in w_2:
        L.append(10*np.log10(((0.2407*i**2)**2+(12.04*i)**2)/((1-39.2*i**2)**2+(-12.25*i**3+12.25*i)**2)))
        phase.append(atan((i*(2.948575*i**4 + 469.019425*i**2 - 12.04))/(i**2*(138.05456*i**2 - 147.2493)))/np.pi*180-360)
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 8), dpi=100)
    # axes[0].plot(w_, L)
    # axes[1].plot(w_, phase)  # 坐标轴对象axes内置plot函数
    # axes[0].grid(True, linestyle="--", alpha=1)
    # axes[1].grid(True, linestyle="--", alpha=1)
    # axes[0].set_xscale('log')
    # axes[1].set_xscale('log')
    # axes[0].set_xlabel("w")
    # axes[0].set_ylabel("20lg(W)")
    # axes[0].set_title("ЛАЧХ", fontsize=20)
    # axes[1].set_xlabel("w")
    # axes[1].set_ylabel("Фаза")
    # axes[1].set_title("ЛФЧХ", fontsize=20)
    # plt.show()
    index_L_0 = L.index(max(L))
    print('phase[index_L_0]',phase[index_L_0])
    # p = np.array([150.063, 0, 1236.456, 0, -73.299, 0, 1])   # 8
    p = np.array([150.063, 0, 1219.316, 0, -37632.477, 0, 1])
    print(np.roots(p))
    # def func(v):
    #     x, = v.tolist()
    #     return [
    #         150.063 * (x ** 6) + 1236.456 * (x ** 4) - 73.299 * (x ** 2) + 1
    #     ]
    # r = fsolve(func, [0.2])
    # print('r[0]',r[0])
# p45()
# i = 0.1606
# print(10*np.log10(((0.2407*i**2)**2+(12.04*i)**2)/((1-39.2*i**2)**2+(-12.25*i**3+12.25*i)**2)))
# i = 0.17
# print(atan((i*(2.948575*i**4 + 469.019425*i**2 - 12.04))/(i**2*(138.05456*i**2 - 147.2493)))/np.pi*180-180)
# i = 3.505
# print(atan((-47.07*i**5-7479.12*i**3+192*i)/(-2201.472*i**4+2367.36*i**2))*180/np.pi-180)

def p6():
    w_ = []
    for i in range(1, 1000):
        w_.append(i / 1000)
        pass
    u = []
    v = []
    for i in w_:
        u.append(-38.96*i**2+1)
        v.append(-12.25*i**3+0.314*i)
    plt.plot(u,v)
    plt.grid()
    plt.title("Подограф Михайлова")
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()
    pass
p6()