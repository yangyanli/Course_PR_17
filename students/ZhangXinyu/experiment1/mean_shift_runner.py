
import mean_shift as ms
from numpy import genfromtxt

def load_points(filename):
    data = genfromtxt(filename, delimiter=',')
    return data

def run():
    from matplotlib import pyplot as plt
    reference_points = load_points("R15.txt")
    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth = 2)
    mark = ['or', 'ob', 'og', 'ok', '^b', '+g', 'sr', 'db', '<g', 'pr','+b','+r','^g','^r','sb','sg','dr','dg','<r','<b']
    
    for i in range(len(mean_shift_result.shifted_points)):
        original_point = mean_shift_result.original_points[i]
        converged_point = mean_shift_result.shifted_points[i]
        cluster_assignment = mean_shift_result.cluster_ids[i]
        markIndex = int(cluster_assignment)  
        plt.plot(original_point[0], original_point[1], mark[markIndex])
    plt.show()

if __name__ == '__main__':
    run()
