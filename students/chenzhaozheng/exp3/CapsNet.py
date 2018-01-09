# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import mxnet as mx
from mxnet import nd,image,autograd,init
from mxnet.gluon import nn,Trainer
from CapsLayers import DigitCaps, PrimaryConv, Length
from mxnet.gluon.loss import L2Loss

batch_size = 32
epochs = 5
print_batches = 10
lambda_value = 0.5

def load_data_mnist(batch_size, resize=None):
    def transform_mnist(data, label):
        if resize:
            data = image.imresize(data, resize, resize)
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = mx.gluon.data.vision.MNIST(root='./data',
        train=True, transform=transform_mnist)
    mnist_test = mx.gluon.data.vision.MNIST(root='./data',
        train=False, transform=transform_mnist)
    train_data = mx.gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)

def _get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)



def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch, ctx)
        output = net(data)
        acc += nd.mean(nd.argmax(output,axis=1)==label).asscalar()

    return acc / (i+1)

def CapsuleMarginLoss(y_pred,labels,lambda_value):
    #print(y_pred)
    #print(labels)
    labels_onehot = labels 
    first_term_base = nd.square(nd.maximum(0.9-y_pred,0))
    second_term_base = nd.square(nd.maximum(y_pred -0.1, 0))
    
    margin_loss = labels_onehot * first_term_base + lambda_value * (1-labels_onehot) * second_term_base
    margin_loss = margin_loss.sum(axis=1) 
    return margin_loss
    

def CapsNet(batch_size, ctx):

    net = nn.Sequential()
    with net.name_scope():

        net.add(nn.Conv2D(channels=256, kernel_size=9, strides=1, padding=(0,0), activation='relu'))
        net.add(PrimaryConv(dim_vector=8, n_channels=32, kernel_size=9, strides=2, context=ctx, padding=(0,0)))
        net.add(DigitCaps(num_capsule=10, dim_vector=16, context=ctx))
        net.add(Length())

    net.initialize(ctx=ctx, init=init.Xavier())
    return net


if __name__ == "__main__":
    ctx = mx.cpu()

    train_data, test_data = load_data_mnist(batch_size=batch_size,resize=28)
    #print(train_data.shape)
    net = CapsNet(batch_size=batch_size,ctx=ctx)
    print(net)
    trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.01})

    for epoch in range(epochs):
        train_loss0 = 0.
        train_acc0 = 0.
        train_loss = 0.
        train_acc = 0.
        for i, batch in enumerate(train_data):
            data, label = batch
            one_hot_label = nd.one_hot(label,10)

            label = label.as_in_context(ctx)
            one_hot_label = one_hot_label.as_in_context(ctx)
            data = data.as_in_context(ctx)
            
            with autograd.record():
                output = net(data)
                L = CapsuleMarginLoss(output, one_hot_label,lambda_value)
            
            L.backward()
            trainer.step(data.shape[0])
            n = i+1

            train_loss += nd.mean(L).asscalar()
            train_acc += nd.mean(nd.argmax(output,axis=1)==label).asscalar()
            if i%20==19:
                print("Epoch %d. Batch %d-%d. Loss: %f, Train acc %f"\
                 % (epoch+1,i-19,i, (train_loss-train_loss0)/20, (train_acc-train_acc0)/20))
                train_loss0 = train_loss
                train_acc0 = train_acc

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (epoch+1, train_loss/n, train_acc/n, test_acc))
        net.collect_params().save(filename='params_net')
