import math
import random
import numpy as np
import matplotlib.pyplot as plt

class meanshift():
    maxn = 10000
    r = 2
    theta = 0.5
    points = np.zeros([maxn, 3], dtype=np.float32)
    ori = np.zeros([maxn, 3], dtype=np.float32)
    points_num = 0
    prefix = "D:\OneDrive\Document\Learning\Crouse\Junior\Pattern Recognition\Clustering\synthetic_data\\%s"

    def load_data(self):
        file_in = open(self.prefix % 'flame.txt')
        c = 0
        for row in file_in:
            row = row.strip('\n')
            point = row.split(",")
            self.points[c][0:2] = point[0:2]
            self.ori[c] = point
            self.points[c][2] = c
            c += 1
        self.points_num = c
        print("There are ", c, " points\n")
        file_in.close()

    def weightGaussian(self, d):
        return 1


    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

    def output(self):
        file_out = open(self.prefix % 'flame_r3t0.5.txt',"w")
        for i in range(self.points_num):
            file_out.write("{},{},{}\n".format(self.ori[i,0],self.ori[i,1],self.points[i][2]))
        file_out.close()

    def shift(self):
        t = 0
        while True:
            maxmove = 0
            for i in range(self.points_num):
                x = self.points[i][0]
                y = self.points[i][1]
                c = self.points[i][2]
                sumx = sumy = counter = 0
                for j in range(self.points_num):
                    dis = self.distance(x, y, self.points[j][0], self.points[j][1])
                    if dis <= self.r:
                        w = self.weightGaussian(dis)
                        sumx += (self.points[j][0] - x)*w
                        sumy += (self.points[j][1] - y)*w
                        self.points[j][2] = c
                        counter += 1
                sumx /= counter
                sumy /= counter
                self.points[i][0] += sumx
                self.points[i][1] += sumy
                tmp = abs(sumx) + abs(sumy)
                if(maxmove < tmp):
                    maxmove = tmp
            t += 1
            print(t,"th loop, maxmove =", maxmove)
            if (maxmove < self.theta):
                break

    def clear(self):
        print()


if __name__ == "__main__":
    ms = meanshift()
    ms.load_data()
    ms.shift()
    ms.output()

    fig = plt.figure(figsize=(10,5))
    img0 = fig.add_subplot(121)
    plt.scatter(ms.ori[:,0], ms.ori[:,1], c = ms.ori[:,2])
    img0.set_title("ORI")
    img1 = fig.add_subplot(122)
    plt.scatter(ms.ori[:,0], ms.ori[:,1], c = ms.points[:,2])
    img1.set_title("meanshift")

    plt.show()
