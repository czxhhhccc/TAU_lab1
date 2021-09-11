import control.matlab as matlab
import numpy as np
import matplotlib.pyplot as plt
import math
import colorama as color
import control

def choice2():
    print(color.Style.RESET_ALL)
    inertialessUnitName = 'Безынерционное звено'
    aperiodicUnitName= 'Апериодическое звено'
    integratingUnitName = 'Интегрирующее звено'
    idealdifferentialUnitName = 'Идеальное дифференцирующее звено'
    realdifferentialUnitName = 'Реальное дифференцирующее звено'

    needNewChoice = True
    while needNewChoice:
        print(color.Style.RESET_ALL)
        userInput=input('Введите номер команды: \n'
                        '1-' +inertialessUnitName+';\n'
                        '2-'+aperiodicUnitName+';\n'
                        '3-'+integratingUnitName+';\n'
                        '4-'+idealdifferentialUnitName+';\n'
                        '5-'+realdifferentialUnitName+'.\n')
        if userInput.isdigit():
            needNewChoice = False
            userInput=int(userInput)
            if userInput==1:
                name = 'Безынерционное звено'
                pass
            elif userInput==2:
                name='Апериодическое звено'
                pass
            elif userInput==3:
                name='Интегрирующее звено'
                pass
            elif userInput==4:
                name='Идеальное дифференцирующее звено'
                pass
            elif userInput==5:
                name='Реальное дифференцирующее звено'
                pass
            else:
                print('недопустимое значение')
                needNewChoice=True
            pass
        else:
            print(color.Fore.LIGHTBLUE_EX+'\nПожалуйста, введите чиловое значение!')
            needNewChoice = True
            pass
    return name
    pass
def getUnit(name):
    needNewChoice = True
    while needNewChoice:
        needNewChoice = False
        k = input('пожалуйста, введите коэф. "k"')
        T = input('пожалуйста, введите коэф. "T"')
        if k.isdigit() and T.isdigit():
            k = int(k)
            T = int(T)
            if name == 'Безынерционное звено':
                unit = matlab.tf([k],[1])
                pass
            elif name == 'Апериодическое звено':
                unit = matlab.tf([k],[T,1])
                pass
            elif name == 'Интегрирующее звено':
                unit = matlab.tf([k],[1,0])
                pass
            elif name == 'Идеальное дифференцирующее звено':
                unit = matlab.tf([k,0],[T,1])
                pass
            elif name == 'Реальное дифференцирующее звено':
                unit = matlab.tf([k,0],[T,1])
                pass
        else:
            print(color.Fore.LIGHTBLUE_EX+'\nПожалуйста, введите чиловое значение!')
            needNewChoice = True
            pass
        pass
    return unit
def graph1(num, title,y,x):
    plt.subplot(4,1,num)
    plt.grid(True)
    if title=='переходная характеристика':
        plt.plot(x,y,'purple')
        plt.ylabel('амплитуда')
        plt.xlabel("Время c")
        pass
    elif title=='импульная характеристика':
        plt.plot(x,y,'green')
        plt.ylabel('амплитуда')
        plt.xlabel("Время c")
        pass
    elif title=='АЧХ':
        plt.plot(x,y,'purple')
        plt.ylabel('амплитуда')
        plt.xlabel('Частота, rad/s')
        pass
    elif title=='ФЧХ':
        plt.plot(x,y,'green')
        plt.ylabel('Фаза')
        plt.xlabel("Частота, rad/s")
        pass
    plt.title(title)
    pass
unitName = choice2()
unit = getUnit(unitName)
print(unit)
timeLine = []
for i in range(1,10000):
    timeLine.append(i/1000)
    pass
if unitName=='Идеальное дифференцирующее звено':
    print('При идуальном дифференцирующем звене переходная характеристика представляет сщбой переходная функциф')
    pass
else:
    [y,x]=matlab.step(unit,timeLine)
    graph1(1,'переходная характеристика',y,x)
    [y,x]=matlab.impulse(unit,timeLine)
    graph1(2,'импульная характеристика',y,x)
    pass
mag, phase, omega=control.bode(unit, plot=False)
graph1(3,'АЧХ',mag,omega)
phase1=[]
if unitName=='Реальное дифференцирующее звено':
    for i in phase:
        phase1.append(i+2 * np.pi)
        pass
    graph1(4, 'ФЧХ', phase1, omega)
    pass
else:
    graph1(4, 'ФЧХ', phase, omega)
    pass
plt.show()





