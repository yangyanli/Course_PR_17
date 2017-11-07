import matplotlib.pyplot as plt
import numpy as np
import random
import os


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm((x[0] - mu[i[0]][0], x[1] - mu[i[0]][1]))) for i in enumerate(mu)],
                        key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return (mu, clusters)


def main():
    parr = []
    carr = []

    with open(os.path.join('synthetic_data', 'Aggregation.txt'), 'r') as f:
        s = f.readlines()
        for l in s:
            x, y, c = map(lambda x: float(x), l.strip().split(','))
            parr.append((x, y))
            carr.append(c)
    mu, clusters = find_centers(parr, 7)
    print(len(clusters))
    for _, i in clusters.items():
        color = [random.random() for _ in range(3)]
        for j in i:
            plt.scatter(*j, c=color, marker='.')
    plt.show()


if __name__ == '__main__':
    main()
