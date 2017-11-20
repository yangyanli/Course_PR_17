from collections import defaultdict
from random import uniform
from math import sqrt
from draw_data import draw
from random import choice


def point_avg(points):

    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):

    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means:
        centers.append(point_avg(new_means[points]))

    return centers

def update_centers_plus(data_set,assignments):

    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means:
        centers.append(point_avg_plus(new_means[points]))

    return centers

def point_avg_plus(points):

    delta = []


    for i in range(len(points)):
        the_point = points[i]
        points.remove(the_point)

        temp = 0

        for p in points:
            temp += manhadun(p,the_point)
        delta.append(temp)

        points.append(the_point)

    min_e = min(delta)
    index = delta.index(min_e)

    return points[index]


def manhadun(p1,p2):

    demansion = len(p1)

    p = 0.0

    for i in range (demansion):
        p += abs(p1[i]-p2[i])

    return p
def assign_points(data_points, centers):

    assignments = []
    for point in data_points:
        shortest = 100000  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):

    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers




def generate_k_plus(dataset,k):

    centers = []
    first = choice(dataset)
    dataset.remove(first)
    centers.append(first)


    for i in range(k-1):
        datadx = []
        for data in range(len(dataset)):
            minval = 10000
            for cen in centers:
                val = distance(cen, dataset[data])
                if val<minval:
                    minval = val

            datadx.append(minval)

        for k in range(1,len(datadx)):
            datadx[k] = datadx[k]+datadx[k-1]

        a  = uniform(0,datadx[-1])
        index_ = 0
        for i in range(len(datadx)):
            if datadx[i]>a:
                index_= i
                break
        centers.append(dataset[index_])
        dataset.remove(dataset[index_])

    return centers










def k_means_plus(filename,k):

    dataset = get_data_set(filename)
    fileout = open('result/kmeans_plus-' + filename[7:-4] + '.txt', 'w')
    k_points = generate_k_plus(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)

    for i in range(len(assignments)):
        fileout.write(str(dataset[i][0]) + ',' + str(dataset[i][1]) + ',' + str(assignments[i]) + '\n')


def k_means(filename, k):

    dataset = get_data_set(filename)
    fileout = open('result/kmeans-'+filename[7:-4]+'.txt','w')
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)

    for i in range(len(assignments)):
        fileout.write(str(dataset[i][0])+','+ str(dataset[i][1])+','+str(assignments[i])+'\n')


def k_means_plus_medoids(filename, k):

    dataset = get_data_set(filename)
    fileout = open('result/kmeans_plus_medoids-' + filename[7:-4] + '.txt', 'w')
    k_points = generate_k_plus(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    count = 0
    while assignments != old_assignments:
        new_centers = update_centers_plus(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        count+=1
        if count>1000:
            break

    for i in range(len(assignments)):
        fileout.write(str(dataset[i][0]) + ',' + str(dataset[i][1]) + ',' + str(assignments[i]) + '\n')

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

    return dataset




if __name__ == '__main__':
    #agg 7 flame 2 mix 23 r15 15

    # file = 'ordata/Aggregation.txt'
    # file = 'ordata/flame.txt'
    file = 'ordata/mix.txt'
    # file = 'ordata/R15.txt'


    k_means_plus_medoids(file,23)
    # k_means_plus(file, 23)
    #
    draw('result/kmeans_plus_medoids-'+file[7:-4]+'.txt')
    # draw('result/kmeans_plus-' + file[7:-4] + '.txt')


