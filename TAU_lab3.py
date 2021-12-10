from control import pzmap, bode
import control.matlab as matlab
import numpy as np
import matplotlib.pyplot as plt
import colorama as color
from sympy import symbols,poly, simplify, re, im
from math import sqrt,atan,fabs, exp, log10
import mpmath as mp
from decimal import Decimal

def Step_no_PID():
    W1 = matlab.tf([1], [7, 1])  # генератор
    W2 = matlab.tf([0.02, 1], [0.35, 1])  # турбина
    W3 = matlab.tf([24], [5, 1])  # усилитель
    W_ForwardGain = W1*W2*W3
    W_cc = matlab.feedback(W_ForwardGain, 1, sign=-1)
    print(W_cc)
    timeLine = []
    for i in range(0, 5000):
        timeLine.append(i / 20)
        pass
    [y, x] = matlab.step(W_cc, timeLine)
    plt.plot(x,y)
    plt.grid()
    plt.show()
# Step_no_PID()

def PID(k_P, T_I, T_D):
    W1 = matlab.tf([1],[7,1])            # генератор
    W2 = matlab.tf([0.02,1],[0.35,1])    # турбина
    W3 = matlab.tf([24],[5,1])           # усилитель
    W_P = matlab.tf([k_P],[1])
    W_I = matlab.tf([1],[T_I,0])
    W_D = matlab.tf([T_D,0],[1])
    # print('W_P',W_P)
    # print('W_I', W_I)
    # print('W_D', W_D)
    W4 = W_P+W_I+W_D
    # print(W4)
    W_PID_ForwardGain = W1*W2*W3*W4
    W_PID = matlab.feedback(W_PID_ForwardGain,1,sign=-1)
    W_PID_Open = W1*W2*W3*W4
    # print("W_PID",W_PID)
    # W1 = matlab.tf([8, 0], [1])  # -0.501482050306848
    # W2 = matlab.tf([1], [7, 1])
    # W3 = matlab.tf([0.02, 1], [0.35, 1])
    # W4 = matlab.tf([24], [5, 1])
    # W_ForwardGain = W2 * W3 * W4
    # W_OpenLoop = W1 * W2 * W3 * W4
    # # W_ClosedLoop = W_OpenLoop/(1+W1*W_OpenLoop)   # 数值带入
    # W_ClosedLoop = matlab.feedback(W_ForwardGain, W1, sign=-1)

    # print('W_P',W_P)
    # print('W_I',W_I)
    # print('W_D',W_D)
    # W_ForwardGain = W2*W3*W4
    # W_OpenLoop = W1*W2*W3*W4
    # W_ClosedLoop = W_OpenLoop/(1+W1*W_OpenLoop)   # 数值带入
    # W_ClosedLoop = matlab.feedback(W_ForwardGain,W1,sign=-1)
    # print('W_OpenLoop составляет',W_OpenLoop)
    # print('W_ForwardGain составляет',W_ForwardGain)
    # print('W_ClosedLoop',W_ClosedLoop)
    # return W_ClosedLoop
    return W_PID, W_PID_Open
def PI(k_P, T_I):
    W1 = matlab.tf([1],[7,1])            # генератор
    W2 = matlab.tf([0.02,1],[0.35,1])    # турбина
    W3 = matlab.tf([24],[5,1])           # усилитель
    W_P = matlab.tf([k_P],[1])
    W_I = matlab.tf([1],[T_I,0])
    W4 = W_P+W_I
    W_PI_ForwardGain = W1*W2*W3*W4
    W_PI = matlab.feedback(W_PI_ForwardGain,1,sign=-1)
    W_PI_Open = W1 * W2 * W3 * W4
    # print(W_PI)
    return W_PI, W_PI_Open

# PID
def test1_PID():
    kp = np.linspace(0.1,3,100)
    for ii in kp:
        plt.clf()
        timeLine = []
        for i in range(0, 1000):
            timeLine.append(i / 10)
            pass
        W1 = PID(ii,3,2)
        [y, x] = matlab.step(W1, timeLine)
        plt.plot(x,y)
        plt.plot([11,11],plt.ylim(), 'r--')
        plt.plot(plt.xlim(), [0.95, 0.95], 'r--')
        plt.plot(plt.xlim(), [1.05, 1.05], 'r--')
        plt.title('iter:{}'.format(str(ii)))  # title 可以直观展现出  迭代进行到哪一步了
        plt.grid()
        plt.pause(0.001)
        # plt.show()
        pass
# test1()
def test2_PID():
    Ti = np.linspace(100,200,150)
    for ii in Ti:
        plt.clf()
        timeLine = []
        for i in range(0, 1000):
            timeLine.append(i / 10)
            pass
        W1 = PI(0.076,ii)
        [y, x] = matlab.step(W1, timeLine)
        plt.plot(x,y)
        plt.plot([11,11],plt.ylim(), 'r--')
        plt.plot(plt.xlim(), [0.95, 0.95], 'r--')
        plt.plot(plt.xlim(), [1.05, 1.05], 'r--')
        plt.title('iter:{}'.format(str(ii)))  # title 可以直观展现出  迭代进行到哪一步了
        plt.grid()
        plt.pause(0.001)
        # plt.show()
        pass
# test2()
W_PID,W_PID_Open = PID(1,20,2)   #1,20,2
def Step(W):
    timeLine = np.linspace(0, 200, 2000)
    [y, x] = matlab.step(W, timeLine)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    # plt.yticks(yticks)
    # ax.set_xlabel('t, c')
    # ax.set_ylabel('h(t)')
    ax.grid()
    yticks = np.linspace(0, 1.5, 16)
    plt.yticks(yticks)
    # xticks = np.linspace(0, 700, 8)
    # plt.xticks(xticks)
    x = list(x)
    y = list(y)
    y_max_index = y.index(max(y))  # get the index of the maximum inflation
    y_max = y[y_max_index]  # get the inflation corresponding to this index
    x_max = x[y_max_index]  # get the year corresponding to this index
    y_stable = y[len(y) - 150]
    y_ReverseOrder = list(np.flipud(y))
    for i in y_ReverseOrder:
        if i > 1.05 * y_stable or i < y_stable * 0.95:
            y_regulation_index = len(y) - y_ReverseOrder.index(i)
            # y_regulation_index = y.index(i)
            x_regulation = x[y_regulation_index]  # врекмя регулирования
            y_regulation = y[y_regulation_index]
            break
    # print(y_max_index)
    text = 'x={:.3f}, y={:.3f}'.format(x_max, y_max)
    ax.annotate(text, xy=(x_max, y_max), xytext=(x_max, y_max + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.01))
    text1 = 'x={:.3f}, y={:.3f}'.format(x[len(y) - 150], y[len(y) - 150])
    ax.annotate(text1, xy=(x[len(y) - 150], y[len(y) - 150]), xytext=(x[len(y) - 150], y[len(y) - 150] + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.01))
    text2 = 'x={:.4f}, y={:.4f}'.format(x_regulation, y_regulation)
    ax.annotate(text2, xy=(x_regulation, y_regulation), xytext=(x_regulation, y_regulation + 0.1),
                arrowprops=dict(facecolor='red', shrink=0.01))
    plt.title(f'Переходная характеристика\n'
              'перерегулирование:{:.3f}%\n'.format((y_max - y[len(y) - 150]) / y[len(y) - 150] * 100) +
              'время регулирования:{:.3f}'.format(x_regulation)+'c')
    plt.show()
    pass
def Step1(W):
    x = []
    for i in range(1, 1000):
        x.append(i / 20)
        pass
    y = []
    num = W.num[0][0]
    den = W.den[0][0]
    len_num = len(num)
    len_den = len(den)
    f1 = lambda p: sum(coef * p ** (len_num - i - 1) for i, coef in enumerate(num)) / (
                sum(coef * p ** (len_den - i - 1) for i, coef in enumerate(den)) * p)
    for i in x:
        y.append(mp.invertlaplace(f1, i))
    plt.plot(x,y)
    plt.grid()
    plt.show()
def M(W):
    num = np.flipud(W.num[0][0])
    den = np.flipud(W.den[0][0])
    p = symbols('p')
    w = symbols('w',real = True)
    f1 = poly(sum(coef*p**i for i,coef in enumerate(num)))/poly(sum(coef*p**i for i,coef in enumerate(den)))
    f2 = f1.subs(p,complex(0, 1)*w)
    print(f1)
    f2_real = simplify(re(f2))
    f2_imag = simplify(im(f2))
    f2_ = []
    x = []
    for i in range(1, 1000):
        x.append(i / 100)
        pass
    for i in x:
        f2_.append(sqrt(f2_real.subs(w,i)**2+f2_imag.subs(w,i)**2))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f2_)
    # plt.yticks(yticks)
    # ax.set_xlabel('t, c')
    # ax.set_ylabel('h(t)')
    ax.grid()
    yticks = np.linspace(0, 1.5, 16)
    plt.yticks(yticks)
    # xticks = np.linspace(0, 700, 8)
    # plt.xticks(xticks)
    x = list(x)
    y = f2_
    y = list(y)
    y_max_index = y.index(max(y))  # get the index of the maximum inflation
    y_max = y[y_max_index]  # get the inflation corresponding to this index
    x_max = x[y_max_index]  # get the year corresponding to this index
    # print(y_max_index)
    text = 'x={:.3f}, y={:.3f}'.format(x_max, y_max)
    ax.annotate(text, xy=(x_max, y_max), xytext=(x_max, y_max + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.01))
    text1 = 'x={:.3f}, y={:.3f}'.format(x[0], y[0])
    ax.annotate(text1, xy=(x[0], y[0]), xytext=(x[0], y[0] + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.01))
    M_ = y_max / y[0]
    plt.title('АЧХ показатель колебательности:{:.3f}'.format(M_))
    plt.show()
    return x, f2_, M_
# Step(W_PID)
# Step1(W_PID)
# M(W_PID)

#PI
def test1_PI():
    kp = np.linspace(0.061,0.081,150)
    for ii in kp:
        plt.clf()
        timeLine = []
        for i in range(0, 1000):
            timeLine.append(i / 10)
            pass
        W1 = PI(ii,160)
        [y, x] = matlab.step(W1, timeLine)
        plt.plot(x,y)
        plt.plot([11,11],plt.ylim(), 'r--')
        plt.plot(plt.xlim(), [0.95, 0.95], 'r--')
        plt.plot(plt.xlim(), [1.05, 1.05], 'r--')
        plt.title('iter:{}'.format(str(ii)))  # title 可以直观展现出  迭代进行到哪一步了
        plt.grid()
        plt.pause(0.001)
        # plt.show()
        pass
# test1()
def test2_PI():
    Ti = np.linspace(100,200,150)
    for ii in Ti:
        plt.clf()
        timeLine = []
        for i in range(0, 1000):
            timeLine.append(i / 10)
            pass
        W1 = PI(0.076,ii)
        [y, x] = matlab.step(W1, timeLine)
        plt.plot(x,y)
        plt.plot([11,11],plt.ylim(), 'r--')
        plt.plot(plt.xlim(), [0.95, 0.95], 'r--')
        plt.plot(plt.xlim(), [1.05, 1.05], 'r--')
        plt.title('iter:{}'.format(str(ii)))  # title 可以直观展现出  迭代进行到哪一步了
        plt.grid()
        plt.pause(0.001)
        # plt.show()
        pass
# test2()
W_PI, W_PI_Open = PI(0.0781,164.2)   #0.0781,164.2   11.3   且当timeLine = np.linspace(0, 200, 6000)
Step(W_PI)
# Step1(W_PI)
# M(W_PI)

def Direct_Assess(W):
    '''
    direct assessments of the quality of the transient process
    :return:
    '''
    timeLine = np.linspace(0,200,2001)
    [y,x] = matlab.step(W,timeLine)
    y = list(y)
    x = list(x)
    y_max_index = y.index(max(y))
    y_max = y[y_max_index]
    x_max = x[y_max_index]
    y_stable = y[len(y)-150]
    # время регулирования
    y_ReverseOrder = list(np.flipud(y))
    for i in y_ReverseOrder:
        if i > 1.05 * y_stable or i < y_stable * 0.95:
            y_regulation_index = len(y)-y_ReverseOrder.index(i)
            # y_regulation_index = y.index(i)
            x_regulation = x[y_regulation_index]  # врекмя регулирования
            y_regulation = y[y_regulation_index]
            break
    print('Время регулирования: {:.3f}'.format(x_regulation)+'с')
    # переругилирование
    Ase2 = (y_max-y_stable)/y_stable*100
    print('Переругилирование: {:.3f}'.format(Ase2)+'%')
    # Колебательность
    n = 0
    index = range(0,y_regulation_index)
    for i in index:
        if y[i]<1 and y[i+1]>1:
            n = n+1
        elif y[i]>1 and y[i+1]<1:
            n = n + 1
    print('Колебательность: {}'.format(Decimal(n).quantize(Decimal("1"), rounding = "ROUND_HALF_UP")))
    # Степень затузания
    left_index = []
    index = range(0, y_regulation_index)
    for i in index:
        if y[i] < 1 and y[i + 1] > 1:
            left_index.append(i)
    count = 0
    A_max = []
    for i in left_index:
        count+=1
        y_ = y[i:]
        A_max.append(max(y_))
        if count == 2:
            break
    print(len(A_max))
    if len(A_max) <= 1:
        print('Степени затухания не существует')
    else:
        print('Степень затухания: {:.3f}'.format(1-A_max[0]/A_max[1]))
    # Величину и время достижения первого максимума
    print('Величина первого максимума: {:.3f}'.format(y_max))
    print('Время достижения первого максимума: {:.3f}'.format(x_max)+'c')
print('Определить прямые оценки качества переходного процесса:\n')
Direct_Assess(W_PI)

# По распределению корней на комплексной плоскости замкнутой САУ
def distributionofroots(W):
    Poles,Zeros = pzmap(W,plot=True,grid=False)
    plt.show()
    print(Poles)
    print(Zeros)
    Poles_real = [re(i) for i in Poles]
    Poles_imag = [im(i) for i in Poles]
    # Минимальная величина вещественной части корней
    min_real = -max(Poles_real)
    angle_list = []
    index_roots = range(0,len(Poles))
    for i in index_roots:
        if Poles_imag[i] == 0:
            continue
        else:
            angle_list.append(atan(fabs(Poles_imag[i]/Poles_real[i])))
    max_angle = max(angle_list)
    print(max_angle)
    # Время регулирования (время переходного процесса)
    print('Время регулирования: {:.3f}'.format(3/min_real) + 'с')
    # Перерегулирование
    print('Перерегулирование меньше {:.3f}'.format(exp(-np.pi/max_angle)*100)+'%')
    # Степень колебательности
    print('Степень колебательности {:.3f}'.format(max_angle))
    # Степень затухания
    print('Степень затухания {:.3f}'.format(1-exp(-2*np.pi/max_angle)))
print('По распределению корней на комплексной плоскости замкнутой САУ определить:\n')
distributionofroots(W_PI)
print(W_PID)


def Logarithmic_characteristics(W,omega_limits = [0.001,10000]):
    bode(W,dB = True,plot =True,omega_limits=omega_limits)  # dB = True指的是画图的时候单位为分贝
    plt.show()
def solve_w_c(n,W, omega_limits = [0.001,10000]):
    '''
    Для определения запаса по фазе
    в общем случае запас по фазе существует
    :param omega_limits:
    :return:
    '''
    print(color.Style.RESET_ALL)
    mag, phase, omega = bode(W, dB=True, plot=False,
                                     omega_limits=omega_limits)  # dB = True指的是画图的时候单位为分贝
    # print(phase/np.pi*180)
    # find the cross-over frequency and gain at cross-over
    mag_dB = []
    for i in mag:
        mag_dB.append(20*log10(i))   # bode函数输出的mag只是单纯的幅值 而不是分贝 需要转换
    count1 = 0   # число в первой группе
    for i in np.diff(mag_dB):
        # print(i)
        if i < 0:
            break
        count1 += 1
    if count1 ==0:   # или определить, пуст ли этот набор
        for i in np.diff(mag_dB):
            # print(i)
            if i > 0:
                break
            count1 += 1
    mag_dB1 = []
    mag_dB2 = []
    count2 = 1
    for i in mag_dB:
        if count2 <= count1:
            mag_dB1.append(i)
        else:
            mag_dB2.append(i)
            pass
        count2 += 1
        pass
    omega_magdB1 = []
    omega_magdB2 = []   # разделить омеги на соответствующие группы как амплитуды в дБ
    count3 = 1
    for i in omega:
        if count3 <= count1:
            omega_magdB1.append(i)
        else:
            omega_magdB2.append(i)
            pass
        count3 += 1
        pass
    if np.all(np.diff(mag_dB1)<0):  # Определите, является ли это возрастающей последовательностью
        mag_dB1 = np.flipud(mag_dB1)
        omega_magdB1 = np.flipud(omega_magdB1)
    if np.all(np.diff(mag_dB2)<0):
        mag_dB2 = np.flipud(mag_dB2)
        omega_magdB2 = np.flipud(omega_magdB2)
    # рассчитать запас по фазе
    # Использую интерполяцию для определения угла, когда амплитуда в дБ равно 0
    w_c1 = np.interp(0,mag_dB1,omega_magdB1)
    phase_0_1 = np.interp(w_c1, omega, phase) / np.pi * 180  # 得到的phase是弧度  在波德图上要化成 角度
    margin_phase1 = n*180 + phase_0_1
    if mag_dB2.size>0:   # если вторая группа не пуста
        w_c2 = np.interp(0, mag_dB2, omega_magdB2)
        phase_0_2 = np.interp(w_c2, omega, phase) / np.pi * 180
        margin_phase2 = n*180 + phase_0_2
        # if fabs(margin_phase1) > fabs(margin_phase2):    #неправильно
        if margin_phase1 > margin_phase2 and margin_phase2>0 :
            print(f'запас по фазе равен {margin_phase2}')
            return w_c2, phase_0_2
        elif margin_phase2 > margin_phase1 and margin_phase1>0 :
            print(f'запас по фазе равен {margin_phase1}')
            return w_c1, phase_0_1
    print(f'запас по фазе равен {margin_phase1}')
    return w_c1, phase_0_1
def solve_w_180(n,W,omega_limits = [0.001,10000]):
    '''
    Для определения запаса по модулю
    иногда запаса по модулю не существует
    :param omega_limits:
    :param n: это параметр, который нам нужно регулировать по ЛФЧХ; n=1,2,3...
    :return:
    '''
    mag, phase, omega = bode(W, dB=True, plot=False,
                                     omega_limits=omega_limits)  # dB = True指的是画图的时候单位为分贝
    if np.all(phase>-n*np.pi):   # запас по фазе как правило существует, но запаса по модулю в некотором случае нет
        return 'None', 'None'  # 跳出函数  不再执行下面的语句
    count1 = 0
    for i in np.diff(phase):
        # print(i)
        if i < 0:
            break
        count1 += 1
    if count1 ==0:   # или определить, пуст ли этот набор
        for i in np.diff(phase):
            # print(i)
            if i > 0:
                break
            count1 += 1
    phase1 = []
    phase2 = []
    count2 = 0
    for i in phase:
        if count2 <= count1:
            phase1.append(i)
        else:
            phase2.append(i)
            pass
        count2 += 1
        pass
    omega_phase1 = []
    omega_phase2 = []
    count3 = 0
    for i in omega:
        if count3 <= count1:
            omega_phase1.append(i)
        else:
            omega_phase2.append(i)
            pass
        count3 += 1
        pass
    if np.all(np.diff(phase1)):
        phase1 = np.flipud(phase1)
        omega_phase1 = np.flipud(omega_phase1)
    if np.all(np.diff(phase2)):
        phase2 = np.flipud(phase2)
        omega_phase2 = np.flipud(omega_phase2)
    # рассчитать запас по модулю
    # Использую интерполяцию, чтобы найти амплитуду, когда фаза -180 или -450.
    w_180_1 = np.interp(-n*np.pi, phase1, omega_phase1)
    w_180_2 = np.interp(-n*np.pi, phase2, omega_phase2)
    print(w_180_1)
    print(w_180_2)
    mag_180_1 = np.interp(w_180_1, omega, mag)
    mag_180_2 = np.interp(w_180_2, omega, mag)
    margin_mag1 = -20*log10(mag_180_1)
    margin_mag2 = -20*log10(mag_180_2)
    print('margin_mag2',margin_mag2)
    print('margin_mag1', margin_mag1)
    if margin_mag1<0:
        print(f'запас по модулю равен {margin_mag2}')
        return w_180_2, mag_180_2
    elif margin_mag2<0:
        print(f'запас по моду лю равен {margin_mag1}')
        return w_180_1, mag_180_1
    elif margin_mag1 > margin_mag2:
        print(f'запас по модулю равен {margin_mag2}')
        return w_180_2, mag_180_2
    else:
        print(f'запас по модулю равен {margin_mag1}')
        return w_180_1, mag_180_1
def plot1(w_180, mag_180,n,W, omega_limits = [0.001,10000]):
    '''
    Определить положение соответствующих точек в Л АЧХ и ФЧХ при φ(w) = -(2k+1)*180
    :param w_180:
    :param mag_180:
    :param n:
    :param omega_limits:
    :return:
    '''
    if w_180 == 'None' and mag_180 == 'None':
        print(color.Fore.LIGHTRED_EX+'Для этой системы запаса по модулю не существует')
        return
    # 将目标点定位出来
    bode(W, dB=True, plot=True, omega_limits=omega_limits)  # 画图
    ax1, ax2 = plt.gcf().axes  # get subplot axes
    plt.sca(ax1)  # magnitude plot  对当前轴进操作
    plt.plot(plt.xlim(), [20*log10(mag_180),20*log10(mag_180)], 'r--')
    plt.plot([w_180, w_180], plt.ylim(), 'r--')
    plt.title("Амплитуда при фазе 180 = {0:.3g}".format(mag_180))

    plt.sca(ax2)  # phase plot
    plt.plot(plt.xlim(), [-180*n, -180*n], 'r--')
    plt.plot([w_180, w_180], plt.ylim(), 'r--')
    plt.title("Омега при при фазе 180 = {0:.4g} rad/sec".format(w_180))

    plt.grid()
    plt.show()
def plot2(w_c, phase_0,W, omega_limits = [0.001,10000]):
    '''
    Определить положение соответствующих точек в Л АЧХ и ФЧХ при частоте среза
    :param w_c:
    :param phase_0:
    :param omega_limits:
    :return:
    '''
    # 将目标点定位出来
    bode(W, dB=True, plot=True, omega_limits=omega_limits)
    ax1, ax2 = plt.gcf().axes  # get subplot axes
    # ax1, ax2 = plt.gcf().axes  # get subplot axes
    plt.sca(ax1)  # magnitude plot
    plt.plot(plt.xlim(), [0,0], 'r--')
    plt.plot([w_c, w_c], plt.ylim(), 'r--')
    plt.title("Частота среза = {0:.3g} rad/sec".format(w_c))

    plt.sca(ax2)  # phase plot
    plt.plot(plt.xlim(), [phase_0, phase_0], 'r--')
    plt.plot([w_c, w_c], plt.ylim(), 'r--')
    plt.title("Фаза при частоте = {0:.6g} rad/sec".format(phase_0))

    plt.grid()
    plt.show()
    pass

def log_feature(W):
    Logarithmic_characteristics(W)  # 有内置参数omega_limits = [0.001,1000]
    n = 1  # n =1 при предельном или 3
    w_180, mag_180 = solve_w_180(n, W)
    plot1(w_180, mag_180, n, W)
    w_c, phase_0 = solve_w_c(n, W)
    plot2(w_c, phase_0, W)
# print('По логарифмическим частотным характеристикам определить:')
# log_feature(W_PI_Open)
def Аmplitude_Аrequency(W):
    x, f2_, M_ = M(W)
    print('Показатель колебательности: {:.3f}'.format(M_))
    index_x = range(0,len(x))
    x_slice_index = []
    for i in index_x:
        if i == 0:
            continue
        if f2_[i] > f2_[0] and f2_[i+1]<f2_[0]:
            x_slice_index.append(i)
    x_slice = x[min(x_slice_index)]
    print('Время регулирования: {:.3f}'.format(2*np.pi/x_slice) + 'с')
    print(x_slice_index)
Аmplitude_Аrequency(W_PI)