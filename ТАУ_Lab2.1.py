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

def ChoiceFeedback():
    '''
    выбраются соответственные параметы обратной связи
    :return:
    '''
    HardFeedback = 'Hardfeedback'                            # Ж
    FlexibleFeedback = 'Flexiblefeedback'                    # Г
    AperiodicHardFeedback = 'Aperiodichardfeedback'          # АЖ
    AperiodicFlexibleFeedback = 'Aperiodicflexiblefeedback'  # АГ
    needNewChoice = True
    while needNewChoice:
        print(color.Style.RESET_ALL)
        userInput=input('Введите вид обратной связи: \n'
                        '1-' +HardFeedback+';\n'
                        '2-'+FlexibleFeedback+';\n'
                        '3-'+AperiodicHardFeedback+';\n'
                        '4-'+AperiodicFlexibleFeedback+'.\n')

        if userInput.isdigit():
            needNewChoice = False
            userInput=int(userInput)
            if userInput==1:
                name = HardFeedback
                pass
            elif userInput==2:
                name= FlexibleFeedback
                pass
            elif userInput==3:
                name= AperiodicHardFeedback
                pass
            elif userInput==4:
                name= AperiodicFlexibleFeedback
                pass
            else:
                print('недопустимое значение')
                needNewChoice=True
            pass
        else:
            print(color.Fore.LIGHTBLUE_EX+'\nПожалуйста, введите чиловое и целое значение!')
            needNewChoice = True
            pass
    return name
    pass
def getUnitFeedback(name):
    '''
    получаются соответствующая передаточная функция, коэ. усиления и постоянная времени
    :param name:
    :return:
    '''
    k = float(input('пожалуйста, введите коэф. "k"'))
    T = float(input('пожалуйста, введите коэф. "T"'))
    if name == 'Hardfeedback':
        unit = matlab.tf([k],[1])
        pass
    elif name == 'Flexiblefeedback':
        unit = matlab.tf([k,1],[1])
        pass
    elif name == 'Aperiodichardfeedback':
        unit = matlab.tf([k],[T,1])
        pass
    elif name == 'Aperiodicflexiblefeedback':
        unit = matlab.tf([k,0],[T,1])
        pass
    return unit     #,k,T
def getUnitGenerator():
    '''
    получаются соответствующая передаточная функция
    :param name:
    :return:
    '''
    T = float(input('пожалуйста, введите коэф. "T_Генератор"'))
    unit = matlab.tf([1],[T,1])
    return unit     #,k,T
def ChoiceTurbine():
    '''
    выбраются соответственные параметы турбины
    :return:
    '''
    # UNitGenerator = 'Generator'
    # UnitHydraulicturbine = 'Hydraulic turbine'  # 水轮机
    # UnitSteamturbine = 'Steam turbine'  # 汽轮机
    # UnitAmplifier = 'Amplifier'  # Усилительно – исполнительный орган

    HydraulicTurbine = 'Hydraulic turbine' # гидравлическая турбина
    SteamTurbine = 'Steam turbine' # паровая турбина
    needNewChoice = True
    while needNewChoice:
        print(color.Style.RESET_ALL)
        userInput=input('Введите вид турбины: \n'
                        '1-' +HydraulicTurbine+';\n'
                        '2-'+SteamTurbine+'.\n')

        if userInput.isdigit():
            needNewChoice = False
            userInput=int(userInput)
            if userInput==1:
                name = HydraulicTurbine
                pass
            elif userInput==2:
                name= SteamTurbine
                pass
            else:
                print('недопустимое значение')
                needNewChoice=True
            pass
        else:
            print(color.Fore.LIGHTBLUE_EX+'\nПожалуйста, введите чиловое и целое значение!')
            needNewChoice = True
            pass
    return name
    pass
def getUnitTurbine(name):
    '''
    получаются соответствующая передаточная функция, коэ. усиления и постоянная времени
    :param name:
    :return:
    '''
    if name == 'Hydraulic turbine':
        T_Generator = float(input('пожалуйста, введите коэф. "T_Генератор"'))
        T_HydraulicTurbine = float(input('пожалуйста, введите коэф. "T_Гидравлическая турбины"'))
        unit = matlab.tf([0.01*T_HydraulicTurbine,1],[0.05*T_Generator,1])
        pass
    elif name == 'Steam turbine':
        k_st = float(input('пожалуйста, введите коэф. "k_Паровой турбины"'))
        unit = matlab.tf([k_st],[k_st,1])
        pass
    return unit     #,k,T
def getUnitAmplifier():
    '''
    получаются соответствующая передаточная функция, коэ. усиления и постоянная времени
    :param name:
    :return:
    '''
    k = float(input('пожалуйста, введите коэф. "k_испонительное устройство"'))
    T = float(input('пожалуйста, введите коэф. "T_испонительное устройство"'))
    unit = matlab.tf([k],[T,1])
    return unit     #,k,T
def InPut_():
    nameFeedback = ChoiceFeedback()
    W1 = getUnitFeedback(nameFeedback)
    # print(W1)
    W2 = getUnitGenerator()
    # print(W2)
    nameTurbine = ChoiceTurbine()
    W3 = getUnitTurbine(nameTurbine)
    # print(W3)
    W4 = getUnitAmplifier()
    # print(W4)
    return W1,W2,W3,W4
# 通过外界输入获取传递函数信息
# W1,W2,W3,W4 = InPut_()

W1 = matlab.tf([10,0],[1])
W2 = matlab.tf([1],[7,1])
W3 = matlab.tf([0.02,1],[0.35,1])
W4 = matlab.tf([24],[5,1])
W_ForwardGain = W2*W3*W4
W_OpenLoop = W1*W2*W3*W4
# W_ClosedLoop = W_OpenLoop/(1+W1*W_OpenLoop)   # 数值带入
W_ClosedLoop = matlab.feedback(W_ForwardGain,W1,sign=-1)
print('W_OpenLoop составляет',W_OpenLoop)
print('W_ForwardGain составляет',W_ForwardGain)
print('W_ClosedLoop',W_ClosedLoop)

# переходная характеристика
def Step():
    timeLine = []
    for i in range(0,100000):
        timeLine.append(i/100)
        pass
    [y,x] = matlab.step(W_ClosedLoop,timeLine)
    plt.figure(figsize=(20,8),dpi=100)
    plt.plot(x,y)
    yticks = np.linspace(0, 26, 10)
    plt.title('Переходная характеристика')
    plt.yticks(yticks)
    plt.xlabel('t, c')
    plt.ylabel('h(t)')
    plt.grid()
    plt.show()
    pass
Step()

# корни характеристического уравнения
Root_chara = W_ClosedLoop.pole()
print(Root_chara)
def stability_LocationPole(Root_chara):
    '''
    Оценка устойчивости по виду корней характеристического уравнения.
    :return:
    '''
    Stabel_Sign = True
    # global Root_chara   # тоже можно использовать глобальную переменную
    all_in_left = True
    for i in Root_chara:
        if i.real > 0:
            all_in_left = False
            pass
        pass
    if all_in_left:
        print('Эта система устойчива по виду корней характеристического уравнения!')
        Stabel_Sign = True
        pass
    else:
        print('Эта система не устойчива по виду корней характеристического уравнения!')
        Stabel_Sign = False
    return Stabel_Sign
stability_LocationPole(Root_chara)

# Разомкнуть САУ и оценить устойчивость по критерию Найквиста.
# 只完成了关于开环稳定时的情况   开环不稳定的情况 还没处理
def Nykuist():
    # print(color.Fore.RED+'\nПожалуйста, введите чиловое значение!')
    Root_chara_Open = W_OpenLoop.pole()
    Stabel_Sign_Open = stability_LocationPole(Root_chara_Open)
    print('Stabel_Sign_Open',Stabel_Sign_Open)
    if Stabel_Sign_Open:
        print(color.Fore.RED+'\nОтметите, охватывается ли точка с координатами [-1; j0]')
        real, imag, freq = matlab.nyquist(W_OpenLoop,plot = True)
        # plt.figure(figsize=(20, 8), dpi=60)
        x_ticks = np.linspace(-4,10,15)
        y_ticks = np.linspace(-1,1,3)
        # plt.plot(real, imag)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.grid(True)
        plt.show()
        Sign_embrace = int(input(color.Fore.RED +'1 - охватывается точка с координатами [-1; j0]\n'
                                             '2 - не охватывается точка с координатами [-1; j0]\n'))
        print(color.Style.RESET_ALL)
        if Sign_embrace == 1:
            print('Эта система устойчива')
        elif Sign_embrace == 2:
            print('Эта система не устойчива')
    pass
Nykuist()

# ЛАЧХ и ЛФЧХ разомкнутой системы и определяются запасы по модулю и фазе
def Logarithmic_characteristics(omega_limits = [0.001,10000]):
    mag, phase, omega = control.bode(W_OpenLoop,dB = True,plot =True,omega_limits=omega_limits)  # dB = True指的是画图的时候单位为分贝
    plt.show()
    # find the cross-over frequency and gain at cross-over
    # plt.plot(omega,phase)
    # 用插值求解 角度为180时的幅值
    omega_flipud = False
    if not np.all(np.diff(mag)>0):   # возрастающий или нет
        mag1 = np.flipud(mag)
        omega1 = np.flipud(omega)
        omega_flipud = True
        pass
    if not np.all(np.diff(phase)>0):   # возрастающий или нет
        phase1 = np.flipud(phase)
        if not omega_flipud:
            omega1 = np.flipud(omega)

    # numpy.interp(x, xp, fp, left=None, right=None, period=None)
    # Это функция предназначена для определения омеги при фазе 0 и фмплитуде 0 с пмощью интерполяции
    w_180 = np.interp(-2*np.pi, phase1, omega1)  # 要求xp是递增 fp无所谓 *有所谓xp 和 fp的值得对应
    # phase 里面是弧度
    mag_180 = np.interp(w_180, omega, mag)
    margin_mag = -20*log10(mag_180)
    print(f'w_180={w_180}')
    print(f'mag_180={mag_180}')
    print(f'запас по модулю равен {margin_mag}')
    # 用插值求解赋值为0时的角度
    mag_dB = []
    for i in mag1:
        mag_dB.append(20*log10(i))   # bode函数输出的mag只是单纯的幅值 而不是分贝 需要转换
    # plt.plot(mag_dB,omega)
    plt.show()
    w_c = np.interp(0,mag_dB,omega1)    # 两个必须都倒过来
    # w_c = np.interp(1,mag1,omega1)      # 当然也可以通过幅值为 1 求解
    phase_0 = np.interp(w_c,omega, phase)/np.pi*180    # 得到的phase是弧度  在波德图上要化成 角度
    # print(f'w_c={w_c}')
    print(f'phase_0={phase_0}')
    margin_phase = 180+phase_0
    print(f'запас по фазе равен {margin_phase}')
    # ax1, ax2 = plt.gcf().axes  # get subplot axes

    return w_180, mag_180, w_c, phase_0
def plot1(w_180, mag_180,omega_limits = [0.001,10000]):
    # 将目标点定位出来
    control.bode(W_OpenLoop, dB=True, plot=True, omega_limits=omega_limits)
    ax1, ax2 = plt.gcf().axes  # get subplot axes
    plt.sca(ax1)  # magnitude plot  对当前轴进操作
    plt.plot(plt.xlim(), [20*log10(mag_180),20*log10(mag_180)], 'r--')
    plt.plot([w_180, w_180], plt.ylim(), 'r--')
    plt.title("Амплитуда при фазе 180 = {0:.3g}".format(mag_180))

    plt.sca(ax2)  # phase plot
    plt.plot(plt.xlim(), [-360, -360], 'r--')
    plt.plot([w_180, w_180], plt.ylim(), 'r--')
    plt.title("Омега при при фазе 180 = {0:.4g} rad/sec".format(w_180))

    plt.grid()
    plt.show()
def plot2(w_c, phase_0,omega_limits = [0.001,10000]):
    # 将目标点定位出来
    control.bode(W_OpenLoop, dB=True, plot=True, omega_limits=omega_limits)
    ax1, ax2 = plt.gcf().axes  # get subplot axes

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

# об критерие Найквиста
w_180, mag_180, w_c, phase_0 = Logarithmic_characteristics()     # 有内置参数omega_limits = [0.001,1000]
plot1(w_180, mag_180)
plot2(w_c, phase_0)

# Построить годограф Михайлова. Сделать вывод об устойчивости САУ по критерию Михайлова.
CoefChara = W_ClosedLoop.den[0][0]  #取出第一行 第一列的一维数组
def Mikhailov(CoefChara):
    # 构造闭环特征方程
    # 使用符号系统
    p = symbols('p')
    Chara2_real = 0
    Chara2_imag = 0
    CoefChara_List = list(CoefChara)
    for i in CoefChara_List:
        if (CoefChara_List.index(i) & 1) == 1:
            Chara2_real = Chara2_real + i * (p*complex(0, 1)) ** (len(CoefChara_List) - CoefChara_List.index(i) - 1)
            pass
        if (CoefChara_List.index(i) & 1) == 0:
            Chara2_imag = Chara2_imag + i * (p*complex(0, 1)) ** (len(CoefChara_List) - CoefChara_List.index(i) - 1)
            pass
        pass
    print(Chara2_real)
    print(Chara2_imag)
    # print(Chara2.subs('p','omega*j'))
    # print(Chara2.subs('p','omega*j').real)
    x_Chara2 = np.linspace(0,10,100)
    y_Chara2_real = []
    y_Chara2_imag = []
    for m in x_Chara2:
        y_Chara2_real.append(Chara2_real.subs(p, m))
        y_Chara2_imag.append(Chara2_imag.subs(p, m)/complex(0, 1))
        pass
    plt.plot(y_Chara2_real,y_Chara2_imag)
    plt.title('Годограф Михайлова')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.grid()
    plt.show()
    pass
# 利用米哈伊尔法判定稳定性
Mikhailov(CoefChara)

# критерий Рауса- Гурвица
a = W_ClosedLoop.den[0][0]
def Raus_Hurwitz(a):
    print(color.Style.RESET_ALL)
    a = W_ClosedLoop.den[0][0]
    num_delta = list(range(len(a)))
    del num_delta[0]
    # 只需要求解最大的那个矩阵即可 后期使用切片
    count_ = 0  # 计数变量  每隔两行 左移一列
    dimension = len(a) - 1
    delta = np.zeros((dimension,dimension),dtype=float)  #(3,3)  # zeros from sympy
    # 按 行 给元素赋值
    odd = [x for x in range(dimension + 1) if x % 2 == 1]  # odd 奇数
    even = [x for x in range(dimension + 1) if x % 2 == 0]  # even 偶数
    # print('odd:{}'.format(odd))
    # print('even:{}'.format(even))
    # 只要创建赫尔维兹矩阵即可 后期其子式只要切片即可
    for j in range(dimension):  #[0,1,2]  [0,1] [0]
        if (j+1)%2 == 0: # j=1   # 我要在偶数行赋值
            h = 0
            for m in even:
                delta[j][h+count_] = a[m]
                h+=1
                pass
            pass
        if j%2 == 0: # j=0 2   # 我要在奇数行赋值
            h = 0
            for m in odd:
                delta[j][h+count_] = a[m]
                h += 1
                pass
            pass
        if j%2==1:   # 隔两行往左移一列
            count_ += 1
    print('delta:',delta)
    Stable_sign = True
    CriticallyStable_sign = False
    for kk in num_delta:
        if kk == 1:
            if delta[1,1]<-0.1:   #поскольку погрешность расчета предельного значения kос существует
                Stable_sign = False
                pass
            elif delta[1,1]<=0.1 and delta[1,1]>=-0.1:
                CriticallyStable_sign =True
            pass
        if kk != 1:
            delta_minors = delta[:kk,:kk]# минор  # 切片左开右闭  # 这一步是求取子式
            if det(delta_minors) < -0.1:
                Stable_sign = False
                pass
            if det(delta_minors) >= -0.1 and det(delta_minors) <= 0.1:
                CriticallyStable_sign = True
                pass
            pass
    if Stable_sign and CriticallyStable_sign:
        print('Эта система предельно устойчива')
        pass
    elif Stable_sign:
        print('Эта система устойчива')
    else:
        print('Эта система не устойчива')
    pass
Raus_Hurwitz(a)
