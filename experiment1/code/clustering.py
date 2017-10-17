# -*- coding: utf-8 -*-





if __name__ == "__main__":
    data_file_loc = "../data/mix.txt"
    data_file = open(data_file_loc)
    data = []
    for line in data_file.readlines():
        x, y, t = line.split(",")
        x = float(x)
        y = float(y)
        t = int(t)
        data.append([x,y,t])

