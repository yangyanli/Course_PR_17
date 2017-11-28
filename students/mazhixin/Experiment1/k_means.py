import random
import math
import matplotlib.pyplot as plt
import numpy as np
# m * n * t * d

class kmeans():
    SEED_NUM = 2
    maxn = 10000
    points = np.zeros([maxn, 3],dtype=np.float32)
    seeds = np.zeros([SEED_NUM, 3],dtype=np.float32)
    ori = np.zeros(maxn,dtype=np.float32)
    points_num = 0
    prefix = "D:\OneDrive\Document\Learning\Crouse\Junior\Pattern Recognition\Clustering\synthetic_data\\%s"

    def init(self):
        self.load_data()
        self.scatter()

    def load_data(self):
        file_in = open(self.prefix % 'flame.txt')
        c = 0
        for row in file_in:
            row = row.strip('\n')
            point = row.split(",")
            self.points[c] = point
            self.ori[c] = point[2]
            c += 1
        self.points_num = c
        print("There are ", c, " points\n")
        file_in.close()

    def scatter(self):
        for i in range(self.SEED_NUM):
            ron = random.randint(0, self.points_num)
            self.seeds[i][0:2] = self.points[ron][0:2]
            self.seeds[i][2] = 0

    def update(self):
        change = 0
        seed_sump = np.zeros([self.SEED_NUM, 2], dtype = np.float32)
        while True:
            for i in range(self.points_num):
                dis = 1e8
                index = 0
                res = 0
                #查找最近的种子点
                for j in range(self.SEED_NUM):
                    res = self.distance(self.points[i][0],self.points[i][1],self.seeds[j][0],self.seeds[j][1])
                    if dis > res:
                        dis = res
                        index = j
                self.seeds[index][2] += 1

                #记录种子类别发生变化的数目
                if index != self.points[i][2]:
                    change += 1
                    self.points[i][2] = index


            #终止条件
            if change < 1:
                break
            change = 0

            #更新种子坐标
            s = 0
            for i in range(self.points_num):
                s = int(self.points[i][2])
                seed_sump[s][0] += self.points[i][0]
                seed_sump[s][1] += self.points[i][1]

            for i in range(self.SEED_NUM):
                count = self.seeds[i][2]
                if count > 0:
                    self.seeds[i][0] = seed_sump[i][0] / count
                    self.seeds[i][1] = seed_sump[i][1] / count
                else:
                    self.seeds[i][0] = self.seeds[i][1] = 0

            #清零类体积与坐标和
            for i in range(self.SEED_NUM):
                self.seeds[i][2] = 0
                seed_sump[i][0] = seed_sump[i][1] = 0


    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

    def output(self):
        file_out = open(self.prefix % 'flame_k10.txt',"w")
        for i in range(self.points_num):
            file_out.write("{},{},{}\n".format(self.points[i,0],self.points[i,1],self.points[i][2]))
        file_out.close()


if __name__ == "__main__":

    k = kmeans()
    k.init()
    k.update()
    k.output()

    fig = plt.figure(figsize=(10, 5))
    img0 = fig.add_subplot(121)
    plt.scatter(k.points[:,0], k.points[:,1], c = k.ori)
    img0.set_title("ORI")
    img1 = fig.add_subplot(122)
    plt.scatter(k.points[:,0], k.points[:,1], c = k.points[:,2])
    img1.set_title("K-means")

    plt.show()
