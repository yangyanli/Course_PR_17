import dbscan
import numpy as np
from draw_data import draw


def get_data_set(filename):

    file = open(filename)

    dataset = []
    while True:
        line = file.readline()

        if not line:
            break

        page = line.strip().split(',')

        data = [float(page[0]),float(page[1])]

        dataset.append(data)

    dataset = np.array(dataset)

    dataset = dataset.T


    return dataset
def run(filename):
    data = get_data_set(filename)

    assignments = dbscan.dbscan(data,0.92,5)


    fileout = open('result/dbscan-' + filename[7:-4] + '.txt', 'w')



    dataset = data.T

    for i in range(len(assignments)):
        fileout.write(str(dataset[i][0]) + ',' + str(dataset[i][1]) + ',' + str(assignments[i]) + '\n')

    print(' ')

if __name__ == '__main__':
    #agg 7 flame 2 mix 23 r15 15

    # file = 'ordata/Aggregation.txt'
    # run(file)

    # draw('result/dbscan-' + file[7:-4] + '.txt')

    file = 'ordata/flame.txt'
    run(file)
    draw('result/dbscan-' + file[7:-4] + '.txt')

    # file = 'ordata/mix.txt'
    # run(file)
    #
    # draw('result/dbscan-' + file[7:-4] + '.txt')
    #
    # file = 'ordata/R15.txt'
    # run(file)

    # draw('result/dbscan-' + file[7:-4] + '.txt')