import numpy as np
import matplotlib.pyplot as plt
import math
import random

def init():
    global points, prefix, points_num, eps, \
        category, minPts, neighbor, neighbor_num, raw_num, truth
    maxn = 10000
    prefix = 'data\synthetic_data\\%s'
    points = np.zeros([maxn, 3], dtype=np.float32)
    truth = np.zeros(maxn, dtype=np.float32)
    eps = 1
    minPts = 50
    neighbor = []
    points_num = 0
    category = 1
    neighbor_num = 0
    raw_num = 0


def load_data():
    global points, prefix, points_num, eps, \
        category, minPts, neighbor, neighbor_num, raw_num, truth
    input = open(prefix % 'mix.txt')
    i = 0
    for row in input:
        row = row.strip()
        point = row.split(',')
        points[i] = point
        truth[i] = point[2]
        points[i][2] = 0
        i += 1
    global points_num
    points_num = i
    print('total points number is ', points_num)
    input.close()


def culculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def do(point):
    global points, prefix, points_num, eps, \
        category, minPts, neighbor, neighbor_num, raw_num, truth
    distance = 0
    if point[2] == 0:
        for i in range(points_num):
            distance = culculate_distance(point[0], point[1],
                                          points[i][0], points[i][1])
            if distance <= eps:
                neighbor_num += 1
                if points[i][2] == 0:
                    neighbor.append(i)
                    points[i][2] = category
                    raw_num += 1
        if neighbor_num < minPts:
            for i in range(raw_num):
                neighbor.pop()
        raw_num = 0
        neighbor_num = 0
        while len(neighbor) != 0:
            p = points[neighbor.pop(0)]
            do(p)
        category += 1

def show():
    fig = plt.figure(figsize=(10, 5))
    img0 = fig.add_subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], c=truth)
    img0.set_title("Truth")
    img1 = fig.add_subplot(1, 2, 2)
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2])
    img1.set_title("dbscan")
    plt.show()

def main():
    init()
    load_data()
    for i in range(points_num):
        do(points[i])

    show()

if __name__ == "__main__":
    main()
