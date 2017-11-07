
import meanshift as ms
import numpy as np
from kmeans import get_data_set
from draw_data import draw

def run(file,band):

    dataset = np.array(get_data_set(file))

    meanshifter =  ms.MeanShift()

    result = meanshifter.cluster(dataset, kernel_bandwidth = band)

    fileout = open('result/meanshift-'+file[7:-4]+'.txt','w')

    assignments = result.cluster_ids

    for i in range(len(assignments)):
        fileout.write(str(dataset[i][0])+','+ str(dataset[i][1])+','+str(assignments[i])+'\n')



if __name__ == '__main__':
    # file = 'ordata/Aggregation.txt'
    # run(file)
    # draw('result/meanshift-' + file[7:-4] + '.txt')

    # file = 'ordata/flame.txt'
    # run(file,2)
    # draw('result/meanshift-' + file[7:-4] + '.txt')

    file = 'ordata/mix.txt'
    run(file,1.90)
    draw('result/meanshift-' + file[7:-4] + '.txt')

    # file = 'ordata/R15.txt'
    # run(file,0.6)
    # draw('result/meanshift-' + file[7:-4] + '.txt')