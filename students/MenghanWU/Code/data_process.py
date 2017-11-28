# -*- coding:utf-8 -*-

import colour as colour
import numpy
import random
import matplotlib.pyplot as plt
import matplotlib
import math

a = numpy.loadtxt('flame.txt', str)
# a = numpy.loadtxt('R15.txt', str)

c = []

for colour in a:
    x = colour.split(",")
    x[0] = (float)(x[0])
    x[1] = (float)(x[1])
    x[2] = (int)(x[2])
    c.append(x)

x = []
y = []
z = []
dataset = []
for i in range(0, len(c) - 1):
    x.append(c[i][0])
    y.append(c[i][1])
    z.append(c[i][2])

for i in range(0, len(c) - 1):
    a = [x[i], y[i]]
    dataset.append(a)
data = numpy.array(dataset)
print "==============================================================="
print data
print "==========================================================="


def draw(x, y, z):
    for i in range(len(x)):
        if (z[i] == 0):
            plt.plot(x[i], y[i], 'ro')
        if (z[i] == 1):
            plt.plot(x[i], y[i], 'bo')
        if (z[i] == 2):
            plt.plot(x[i], y[i], 'yo')
        if (z[i] == 3):
            plt.plot(x[i], y[i], 'go')
        if (z[i] == 4):
            plt.plot(x[i], y[i], 'mo')
        if (z[i] == 5):
            plt.plot(x[i], y[i], 'co')
        if (z[i] == 6):
            plt.plot(x[i], y[i], 'ko')
    plt.show()



