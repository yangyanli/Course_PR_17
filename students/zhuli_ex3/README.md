# README

​	实验三我用的是KNN做的MNIST数据集的训练与测试，原理比较简单，代码写起来也不难，最后实现的准确率在93%~95%，不算很高，但是所幸能确定它应该是没有出现bug。

​	这个实验写的时候花的时间不算多，但是实验前的准备才是重中之重。花了一个多星期的时间准备，查询各种非deep learning的方法，包括了形态学处理、SVM等，同时浏览了各种博客，最后选择了KNN，简单高效，运行起来也快，方便进行多次调试。跑完一遍大概用5'10"~5'30"。以下是k取不同值时运行的结果。

## k = 3

running time: 5.00min,20.7377s.
The result:
  error count: 340
  accuracy: 0.9433

##k = 5

running time: 5.00min,26.6690s.
The result:
  error count: 337
  accuracy: 0.9438

## k = 7

running time: 5.00min,14.8880s.
The result:
  error count: 349
  accuracy: 0.9418

## k = 10

running time: 5.00min,14.4516s.
The result:
  error count: 361
  accuracy: 0.9398

## k = 15

running time: 5.00min,12.9438s.
The result:
  error count: 404
  accuracy: 0.9327

## k = 20

running time: 5.00min,14.6250s.
The result:
  error count: 434
  accuracy: 0.9277

## k = 30

running time: 5.00min,27.3815s.
The result:
  error count: 471
  accuracy: 0.9215

​	k值一般会设在20以内，太大会严重影响准确率，该模型在k值为5到10的时候准确率较高。其中，k = 5时准确率为94.38%。