# 模式识别实验三

算法：KNN


* knn.hpp

knn算法头文件，包括算法定义等核心部分

* testMINST.cpp

测试MNIST数据集，对于不同的k(1~25)的准确率，结果保存在test.dat文件中。

* test.dat

文件第n行表示当k=n时10000个MNIST测试数据集中正确匹配的数量，除以10000即为正确率。
由文件可得，当k=3时准确率相对最高(97.05%)，其次为k=7(96.94%)。

* predictSingleNum.cpp

对含有单个数字的图片进行识别，图片和结果截图在predict_singleNum开头的文件夹中。

* predictPhoto.cpp

对含有多个数字的图片进行识别，图片和结果截图在predict_photo开头的文件夹中。