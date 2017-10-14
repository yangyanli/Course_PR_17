# Experiment 1



In this experiment, you need to cluster the datas with your own code.

* Dataset 1 

  The dataset 1 is a synthetic data from [https://cs.joensuu.fi/sipu/datasets/](https://cs.joensuu.fi/sipu/datasets/) and we mix them to one set. You can find the file in "./data/" folder. 

  The expected input file format is a comma separated file, where each row represents a different multi-dimensional data point. If ground truth is provided (and the --no-labels flag isn't specified), the ground truth labels should be placed in the last column of each row.

  For example, the following input file contains 3 data points in 2D, where the first 2 points are part of one cluster and the second one is part of another cluster:

  ```
  0.5, 0.2, 0
  -0.5, 1.0, 0
  10.1, 3.2, 1
  ```

  ​

* Dataset 2

  The datasets is from MNIST [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) , and we use some part of the whole sets. You can 