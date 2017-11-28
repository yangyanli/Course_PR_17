import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.ChangeData("data/mnist.pkl.gz")
net = network.Network([784,30,10])
net.SGD(training_data, 30, 10, 0.5,lmbda=3.0,evaluation_data=test_data,monitor_evaluation_accuracy=True)
net.save("model")


