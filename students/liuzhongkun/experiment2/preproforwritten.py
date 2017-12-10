import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
def removenoise(input):
    #最小最大值滤波3px
    output = input[:][:]
    for i in range(1,len(input) - 1):
        for j in range(1,len(input[0]) - 1):
            minpix = 255
            for k in range(3):
                for l in range(3):
                    if(input[i - 1 + k][j - 1 + k] < minpix):
                        minpix = input[i - 1 + k][j - 1 + k]
            output[i][j] = minpix
    input = output
    for i in range(1, len(input) - 1):
        for j in range(1, len(input[0]) - 1):
            maxpix = 0
            for k in range(3):
                for l in range(3):
                    if (input[i - 1 + k][j - 1 + k] > maxpix):
                        minpix = input[i - 1 + k][j - 1 + k]
            output[i][j] = maxpix
    return output

def image_get(input):

    input_tmp = input[:][:]
    '''
    for i in range(1, len(input_tmp)- 1):
        for j in range(1, len(input_tmp[0])- 1):
            if input_tmp[i][j] > 192:
                if input_tmp[i - 1][j] < 64 or input_tmp[i + 1][j] < 64 or \
                    input_tmp[i][j - 1] < 64 or input_tmp[i][j + 1] < 64:
                    input_tmp[i][j] = 0
    '''
    height = len(input)
    width = len(input[0])
    b = np.zeros([height,width])
    #去背景
    counter = 0
    for i in range(height):
        for j in range(width):
            if input_tmp[i][j] > 128:
                b[i][j] = -1
                counter += 1
    print('counter' + str(counter))

    counter = 1
    domain = []
    for i in range(height):
        for j in range(width):
            if b[i][j] == 0:
                b[i][j] = counter
                keep_domain = [[i,j]]
                stack_tmp = [[i,j]]
                while stack_tmp != []:
                    tmp = stack_tmp[0]
                    stack_tmp = stack_tmp[1:]
                    for m in range(tmp[0]-1, tmp[0] + 2):
                        if m >=0 and m < height:
                            for n in range(tmp[1] - 1, tmp[1] + 2):
                                if n >=0 and n < width and b[m][n] == 0 :
                                    b[m][n] = counter
                                    stack_tmp.append([m,n])
                                    keep_domain.append([m,n])
                domain.append(keep_domain)
                counter += 1

    num = len(domain)
    avg_size = 0
    for i in range(num):
        avg_size += len(domain[i])
    avg_size  = int(avg_size/num)
    i = 0
    each_domain = []
    print(str(avg_size))
    while i < num:
        c_len = len(domain[i])
        if c_len/avg_size > 5 or avg_size/c_len > 5:
            num -= 1
            for noise in domain[i]:
                b[noise[0]][noise[1]] = -1
            domain = domain[0:i]+domain[(i + 1):]
            continue
        size = [10000,10000,-1,-1]
        for j in domain[i]:
            if j[0] > size[2]:
                size[2] = j[0]
            if j[0] < size[0]:
                size[0] = j[0]
            if j[1] > size[3]:
                size[3] = j[1]
            if j[1] < size[1]:
                size[1] = j[1]
        each_domain.append(size)
        i += 1
    # print(b)
    image_out = np.ones([height,width])*255
    for i in range(height):
        for j in range(width):
            if b[i][j] != -1:
                image_out[i][j] = 0
    print(str(num) + ' nums')
    print(str(height) + ' ' + str(width))

    return num, each_domain, image_out

def removerlap(input):
    l = len(input)
    for i in range(l):
        for j in range(i+1, l):
            if input[i][2] < input[j][0] or input[i][3] < input[j][1]:
                continue
            if input[j][2] < input[i][0] or input[j][3] < input[i][1]:
                continue



if __name__ == '__main__':
    im = np.array(Image.open("./test/ex2.jpg"))
    a = np.zeros([len(im),len(im[0])])
    for i in range(len(im)):
        for j in range(len(im[0])):
            a[i][j] = np.mean(im[i][j])

    # a = im[:,:,0].reshape([241,666])
    b = removenoise(a)
    image_get(b)



