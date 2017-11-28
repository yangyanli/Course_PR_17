import numpy as np
import math
import random
import matplotlib.pyplot as plt

global points, seeds, truth, points_num, prefix, SEED_NUM
def init():
    global points, seeds, truth, points_num, prefix, SEED_NUM
    SEED_NUM = 2
    maxn = 10000

    points = np.zeros([maxn, 3], dtype=np.float32)
    seeds = np.zeros([SEED_NUM, 3], dtype=np.float32)
    truth = np.zeros(maxn, dtype=np.float32)
    points_num = 0
    prefix = 'data\synthetic_data\\%s'

def load_data():

    input = open(prefix % 'flame.txt')
    i = 0
    for row in input:
        row = row.strip()
        point = row.split(',')
        points[i] = point
        truth[i] = point[2]
        i += 1
    global points_num
    points_num = i
    print('total points number is ', points_num)
    input.close()

def choose_center():
    for i in range(SEED_NUM):
        cent = random.randint(0, points_num)
        print(cent)
        seeds[i][0:2] = points[cent][0:2]
        seeds[i][2] = 0


def culculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def iterate():
    sum = np.zeros([SEED_NUM, 2], dtype=np.float32)
    count = 0
    change = 0
    while 1:
        count += 1
        for i in range(points_num):
            min_distance = 1e8
            index = 0
            distance = 0;
            for j in range(SEED_NUM):
                distance = culculate_distance(points[i][0], points[i][1],
                                              seeds[j][0], seeds[j][1])
                if distance < min_distance:
                    min_distance = distance
                    index = j
            seeds[index][2] += 1
            if index != points[i, 2]:
                change += 1
                points[i][2] = index
        if change < 1:
            break
        change = 0
        for i in range(points_num):
            category = int(points[i][2])
            sum[category][0] += points[i][0]
            sum[category][1] += points[i][1]
        for i in range(SEED_NUM):
            num = seeds[i][2]
            if num > 0:
                seeds[i][0] = sum[i][0] / num
                seeds[i][1] = sum[i][1] / num
            else:
                seeds[i][0] = seeds[i][1] = 0
        for i in range(SEED_NUM):
            seeds[i][2] = 0
            sum[i][0] = sum[i][1] = 0
    print(count)

def show():
    fig = plt.figure(figsize=(10, 5))
    img0 = fig.add_subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], c = truth)
    img0.set_title("Truth")
    img1 = fig.add_subplot(1, 2, 2)
    plt.scatter(points[:, 0], points[:, 1], c = points[:, 2])
    img1.set_title("K-means")
    plt.show()

def main():
    init()
    load_data()
    choose_center()
    # for i in range(SEED_NUM):
    #     print(seeds[i][0], seeds[i][1], i)
    iterate()
    print(points_num)
    show()

if __name__ == '__main__':

    main()
