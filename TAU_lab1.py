import control.matlab as matlab
import numpy as np
import matplotlib.pyplot as plt
import math
# import color

def chioce():
    inertialessUnitName = 'безынерционное звено'
    aperiodicUnitName= 'Апериодическое звено'

    needNewChoice = True

    while needNewChoice:
        # print(color.)
        userInput=input('Введите номер команды: \n'
                        '1-' +inertialessUnitName+';\n'
                        '2-'+aperiodicUnitName+'.\n')
        if userInput.isdigit():
            needNewChoice = False
            userInput=int(userInput)
            if userInput==1:
                name = 'безынерционное звено'
                pass
            elif userInput==2:
                name='Апериодическое звено'
                pass
            else:
                print('недопустимое значение')
                needNewChoice=True
            pass
        else:
            print('\nПожалуйста, введите чиловое значение!')
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
            if name == 'безынерционное звено':
                unit = matlab.tf([k],[1])
                pass
            elif name == 'Апериодическое звено':
                unit = matlab.tf([k],[T,1])
                pass
        else:
            print('\nПожалуйста, введите чиловое значение!')
            needNewChoice = True
            pass
        pass
    return unit
def graph(num, title,y,x):
    plt.subplot(2,1,num)
    plt.grid(True)
    if title=='переходная характеристика':
        plt.plot(x,y,'purple')
        pass
    elif title=='импульная характеристика':
        plt.plot(x,y,'green')
        pass
    plt.title(title)
    plt.ylabel('амплитуда')
    plt.xlabel("Время c")
unitName = chioce()
unit = getUnit(unitName)
timeLine = []
for i in range(0,10000):
    timeLine.append(i/1000)
    pass
[y,x]=matlab.step(unit,timeLine)
graph(1,'переходная характеристика',y,x)
[y,x]=matlab.impulse(unit,timeLine)
graph(2,'импульная характеристика',y,x)
plt.show()