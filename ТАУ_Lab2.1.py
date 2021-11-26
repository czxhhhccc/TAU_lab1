import control
import control.matlab as matlab
import numpy as np
import matplotlib.pyplot as plt
import colorama as color
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

W1 = matlab.tf([8,0],[1])
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
    # plt.figure(figsize=(20,8),dpi=100)
    plt.plot(x,y)
    # yticks = np.linspace(0, 26, 10)
    plt.title('Переходная характеристика')
    # plt.yticks(yticks)
    plt.xlabel('t, c')
    plt.ylabel('h(t)')
    plt.grid()
    plt.show()
    pass
Step()

# корни характеристического уравнения
Root_chara = W_ClosedLoop.pole()
print('Значения полюсов передаточной функции замкнутой САУ составляет',Root_chara)
def stability_LocationPole(Root_chara):
    '''
    Оценка устойчивости по виду корней характеристического уравнения.
    :return:
    '''
    Stabel_Sign = True
    # global Root_chara   # тоже можно использовать глобальную переменную
    all_in_left = True
    in_virtualAxis = False
    count_rightroot = 0  # для вычисления количества находящих на правой плоскости полюсов
    for i in Root_chara:
        if i.real > 0.01:
            all_in_left = False
            count_rightroot +=1
            pass
        elif i.real >=-0.01 and i.real <=0.01:
            in_virtualAxis = True
        pass
    if all_in_left and in_virtualAxis:
        print('Эта система на границе устойчивости по виду корней характеристического уравнения!')
        Stabel_Sign = True
        pass
    elif all_in_left:
        print('Эта система на границе устойчивости по виду корней характеристического уравнения!')
        Stabel_Sign = True
        pass
    else:
        print('Эта система не устойчива по виду корней характеристического уравнения!')
        Stabel_Sign = False
    return Stabel_Sign,count_rightroot
stability_LocationPole(Root_chara)

# Разомкнуть САУ и оценить устойчивость по критерию Найквиста.
# 只完成了关于开环稳定时的情况   开环不稳定的情况 还没处理
def Nykuist():
    # print(color.Fore.RED+'\nПожалуйста, введите чиловое значение!')
    Root_chara_Open = W_OpenLoop.pole()
    print('Значения полюсов передаточной функции разомкнутой САУ составляет',Root_chara_Open)
    Stabel_Sign_Open,count_rightroot = stability_LocationPole(Root_chara_Open)
    print('Stabel_Sign_Open',Stabel_Sign_Open)
    print(count_rightroot)
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
        Sign_embrace = int(input(color.Fore.LIGHTGREEN_EX +'1 - охватывается точка с координатами [-1; j0]\n'
                                                 '2 - не охватывается точка с координатами [-1; j0]\n'
                                                 '3 - проходит через точку с координатами [-1; j0]\n'   ))
        print(color.Style.RESET_ALL)
        if Sign_embrace == 1:
            print('Эта система не устойчива')
        elif Sign_embrace == 2:
            print('Эта система устойчива')
        elif Sign_embrace == 3:
            print('Эта система на границе устойчивости')
    else:
        positive = float(input(color.Fore.LIGHTBLUE_EX +'Сколько раз охватывала точку с координатами [-1; j0] '
                                                        'при изменении частоты от 0 до плюс бесконечности в положительном напралении\n'
                                             'Положительное направление -- против часовой стрелки\n'))
        if positive == count_rightroot/2:
            print('Эта система устойчива')
        else:
            print('Эта система не устойчива')
    pass
Nykuist()

# ЛАЧХ и ЛФЧХ разомкнутой системы и определяются запасы по модулю и фазе
def Logarithmic_characteristics(omega_limits = [0.001,10000]):
    mag, phase, omega = control.bode(W_OpenLoop,dB = True,plot =True,omega_limits=omega_limits)  # dB = True指的是画图的时候单位为分贝
    plt.show()
def solve_w_c(n,omega_limits = [0.001,10000]):
    '''
    Для определения запаса по фазе
    в общем случае запас по фазе существует
    :param omega_limits:
    :return:
    '''
    print(color.Style.RESET_ALL)
    mag, phase, omega = control.bode(W_OpenLoop, dB=True, plot=False,
                                     omega_limits=omega_limits)  # dB = True指的是画图的时候单位为分贝
    # print(phase/np.pi*180)
    # find the cross-over frequency and gain at cross-over
    mag_dB = []
    for i in mag:
        mag_dB.append(20*log10(i))   # bode函数输出的mag只是单纯的幅值 而不是分贝 需要转换
    count1 = 0
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
    omega_magdB2 = []
    count3 = 1
    for i in omega:
        if count3 <= count1:
            omega_magdB1.append(i)
        else:
            omega_magdB2.append(i)
            pass
        count3 += 1
        pass
    if np.all(np.diff(mag_dB1)<0):
        mag_dB1 = np.flipud(mag_dB1)
        omega_magdB1 = np.flipud(omega_magdB1)
    if np.all(np.diff(mag_dB2)<0):
        mag_dB2 = np.flipud(mag_dB2)
        omega_magdB2 = np.flipud(omega_magdB2)
    # plt.plot(mag_dB1,omega_magdB1)
    # plt.show()
    # plt.plot(mag_dB2, omega_magdB2)
    # plt.show()
    # рассчитать запас по фазе
    # 用插值求解赋值为0时的角度
    w_c1 = np.interp(0,mag_dB1,omega_magdB1)
    # print('mag_dB1',mag_dB1)
    # print('w_c1',w_c1)
    phase_0_1 = np.interp(w_c1, omega, phase) / np.pi * 180  # 得到的phase是弧度  在波德图上要化成 角度
    # print('phase_0_1',phase_0_1)
    margin_phase1 = n*180 + phase_0_1
    if mag_dB2.size>0:
        w_c2 = np.interp(0, mag_dB2, omega_magdB2)
        # print('w_c2', w_c2)
        phase_0_2 = np.interp(w_c2, omega, phase) / np.pi * 180
        # print('phase_0_2', phase_0_2)
        margin_phase2 = n*180 + phase_0_2
        # if fabs(margin_phase1) > fabs(margin_phase2):    #неправильно
        if margin_phase1 > margin_phase2 and margin_phase2>0 :
            print(f'запас по фазе равен {margin_phase2}')
            return w_c2, phase_0_2
        elif margin_phase2 > margin_phase1 and margin_phase1>0 :
            print(f'запас по фазе равен {margin_phase1}')
            return w_c1, phase_0_1
    # print(f'w_c={w_c}')
    # print(f'phase_0_1={phase_0_1}')
    # print(f'phase_0_2={phase_0_2}')
    print(f'запас по фазе равен {margin_phase1}')
    return w_c1, phase_0_1
def solve_w_180(n,omega_limits = [0.001,10000]):
    '''
    Для определения запаса по модулю
    иногда запаса по модулю не существует
    :param omega_limits:
    :param n: это параметр, который нам нужно регулировать по ЛФЧХ; n=1,2,3...
    :return:
    '''
    mag, phase, omega = control.bode(W_OpenLoop, dB=True, plot=False,
                                     omega_limits=omega_limits)  # dB = True指的是画图的时候单位为分贝
    if np.all(phase>-n*np.pi):
        # print(phase>-n*np.pi)
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
    # plt.plot(phase1,omega_phase1)
    # plt.show()
    # plt.plot(phase2, omega_phase2)
    # plt.show()
    # рассчитать запас по модулю
    # 用插值求解相位为-180 或者 -450时的幅值
    w_180_1 = np.interp(-n*np.pi, phase1, omega_phase1)
    w_180_2 = np.interp(-n*np.pi, phase2, omega_phase2)
    mag_180_1 = np.interp(w_180_1, omega, mag)
    mag_180_2 = np.interp(w_180_2, omega, mag)
    print(f'mag_180_1={mag_180_1}')
    print(f'mag_180_2={mag_180_2}')
    margin_mag1 = -20*log10(mag_180_1)
    margin_mag2 = -20*log10(mag_180_2)
    if margin_mag1 > margin_mag2:
        print(f'запас по модулю равен {margin_mag2}')
        return w_180_2, mag_180_2
    else:
        print(f'запас по модулю равен {margin_mag1}')
        return w_180_1, mag_180_1
def plot1(w_180, mag_180,n,omega_limits = [0.001,10000]):
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
    control.bode(W_OpenLoop, dB=True, plot=True, omega_limits=omega_limits)
    # ax1, ax2 = plt.gcf().axes  # get subplot axes
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
def plot2(w_c, phase_0,omega_limits = [0.001,10000]):
    '''
    Определить положение соответствующих точек в Л АЧХ и ФЧХ при частоте среза
    :param w_c:
    :param phase_0:
    :param omega_limits:
    :return:
    '''
    # 将目标点定位出来
    control.bode(W_OpenLoop, dB=True, plot=True, omega_limits=omega_limits)
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

# об критерие Найквиста
Logarithmic_characteristics()     # 有内置参数omega_limits = [0.001,1000]
n = 3  # n =1 при предельном или 3
w_180, mag_180 = solve_w_180(n)
plot1(w_180, mag_180,n)
w_c, phase_0 = solve_w_c(n)
plot2(w_c, phase_0)

CoefChara = W_ClosedLoop.den[0][0]  #取出第一行 第一列的一维数组
def Mikhailov(CoefChara):
    '''
    Построить годограф Михайлова. Сделать вывод об устойчивости САУ по критерию Михайлова.
    :param CoefChara:
    :return:
    '''
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
    # print(Chara2_real)
    # print(Chara2_imag)
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
def Raus_Hurwitz():
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
Raus_Hurwitz()

def k_critical():
    '''
    решить предельное значение коэффициента усиления обратеной свя
    :return:
    '''
    k = symbols('k',real = True)
    p = symbols('p')
    W1_s = (poly(k*p)).as_expr()   # 不加 as_expr() 会报错
    W2_s = 1/poly(7*p+1)
    W3_s = poly(0.02*p+1)/poly(0.35*p+1)
    W4_s = 24/poly(5*p+1)
    W_ForwardGain_s = simplify(W2_s*W3_s*W4_s)
    W_Closedloop = simplify(W_ForwardGain_s/(1+W_ForwardGain_s*W1_s))
    Characteristic_equation = collect(expand(fraction(W_Closedloop)[1]),p)   #  fraction() 提取表达式分子和分母  返回元组
                                                                            # collect() 选取 一个主元进行合并同类项
    Characteristic_equation = Poly(Characteristic_equation,p)   # Poly 是什么作用  为什么有coeffs
    c = Characteristic_equation.coeffs()   # коэффициенты характеристического уравнения
    # print(coeffs_)
    delta3 = Matrix([[c[1],c[3],0],[c[0],c[2],0],[0,c[1],c[3]]])
    delta2 = Matrix([[c[1], c[3]], [c[0], c[2]]])
    delta1 = Matrix([[c[1]]])
    print('delta3:', type((delta3.det())))
    print('delta2:', delta2.det())
    print('delta1:', delta1.det())
    k_range = solve([delta3.det()>=0,delta2.det()>=0,delta1.det()>=0],k)
    print('Диапазон значения k, соответствующие устойчивой системе, составляет',k_range)
    print(type(k_range))
    print(W_Closedloop)
k_critical()
