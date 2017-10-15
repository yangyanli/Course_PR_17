# Experiment 1



In this experiment, you need to cluster the datas with your own code.

![img](http://rubbly.cn/image/pr/exp1_header.png)

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

  The dataset 2 is from MNIST [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) and you can download the dataset from this link. Your task is to cluster the images in "train-images" files. Obvionsly there are 10 classes. It's better if you can visualize the result.

  There are many project about MNIST in github and Internet, so you can find some other method  after you have finished you own task and try to think about which one is better. Good luck.