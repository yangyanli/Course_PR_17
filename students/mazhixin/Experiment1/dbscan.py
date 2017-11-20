import math
import random
import numpy as np
import matplotlib.pyplot as plt

class dbscan():
    maxn = 10000
    points = np.zeros([maxn, 3],dtype=np.float32)
    ori = np.zeros(maxn,dtype=np.float32)
    eps = 1
    minPts = 6
    neighbor = []
    points_num = 0
    cluster_num = 1
    neighbor_num = 0
    neighbor_raw_num = 0
    prefix = "D:\OneDrive\Document\Learning\Crouse\Junior\Pattern Recognition\Clustering\synthetic_data\\%s"

    def load_data(self):
        file_in = open(self.prefix % 'flame.txt')
        c = 0
        for row in file_in:
            row = row.strip('\n')
            point = row.split(",")
            self.points[c] = point
            self.ori[c] = point[2]
            self.points[c][2] = 0
            c += 1
        self.points_num = c
        print("There are ", c, " points\n")
        file_in.close()

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

    def output(self):
        file_out = open(self.prefix % 'flame_e2m12.txt',"w")
        for i in range(self.points_num):
            file_out.write("{},{},{}\n".format(self.points[i,0],self.points[i,1],self.points[i][2]))
        file_out.close()

    def update(self):
        for i in range(self.points_num):
            if self.points[i][2] == 0:
                self.cluster_num += 1
                self.check(self.points[i])

    def check(self, point):
        dis = 0
        for j in range(self.points_num):
            dis = self.distance(point[0], point[1], self.points[j][0], self.points[j][1])
            if dis <= self.eps:
                self.neighbor_num += 1
                if(self.points[j][2] == 0):
                    self.neighbor.append(j)
                    self.points[j][2] = self.cluster_num
                    self.neighbor_raw_num += 1
        if self.neighbor_num < self.minPts:
            for t in range(self.neighbor_raw_num):
                self.neighbor.pop()
        self.neighbor_raw_num = 0
        self.neighbor_num = 0

        while len(self.neighbor) != 0:
            self.check(self.points[self.neighbor.pop(0)])


if __name__ == "__main__":
    db = dbscan()
    db.load_data()
    db.update()
    db.output()

    fig = plt.figure(figsize=(10, 5))
    img0 = fig.add_subplot(121)
    plt.scatter(db.points[:,0], db.points[:,1], c = db.ori * 5)
    img0.set_title("ORI")
    img1 = fig.add_subplot(122)
    plt.scatter(db.points[:,0], db.points[:,1], c = db.points[:,2] * 5)
    img1.set_title("dbscan")

    plt.show()
