import os
import sys
import numpy as np
import tensorflow as tf

from ..input_adaptor.mnist import MnistLoader
from .. import config as cfg
from .capsnet import CapsNet
from matplotlib import pyplot as plt

def make_result_files():
    if not os.path.exists(cfg.result_folder):
        os.mkdir(cfg.result_folder)
    if cfg.is_training:
        loss = str(cfg.result_folder.joinpath('loss.csv'))
        train_acc = str(cfg.result_folder.joinpath('train_acc.csv'))
        val_acc = str(cfg.result_folder.joinpath('val_acc.csv'))

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        return(fd_train_acc, fd_loss)
    else:
        test_acc = str(cfg.result_folder.joinpath('test_acc.csv'))
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train(model):

    fd_train_acc, fd_loss = make_result_files()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    supervisor = tf.train.Supervisor(graph=model.graph, logdir=str(cfg.log_folder), save_model_secs=0)

    with supervisor.managed_session(config=config) as sess:

        supervisor.saver.restore(sess, tf.train.latest_checkpoint(str(cfg.log_folder)))
        for epoch in range(cfg.epoch):
            print('Training ' + str(epoch) + '/' + str(cfg.epoch) + ':')

            if supervisor.should_stop() and False:
                print('Supervisor stoped!')
                break

            loader = MnistLoader()
            num_tr_batch = int(loader.size / cfg.batch_size)

            for step in range(num_tr_batch):
                global_step = epoch * num_tr_batch + step
                print("training step %d/%d." %(step, num_tr_batch))
                labels, imgs = loader.batch(cfg.batch_size)
                imgs = np.array(imgs).astype(np.float32)
                labels = np.array(labels)

                sess.run(model.train_op, {model.X: imgs, model.labels: labels})

                if global_step % 100 == 0:
                    loss = sess.run(model.total_loss, {model.X: imgs, model.labels: labels})
                    train_acc = sess.run(model.accuracy, {model.X: imgs, model.labels: labels})
                    summary_str = sess.run(model.train_summary, {model.X: imgs, model.labels: labels})
                    assert not np.isnan(loss), 'loss is nan!'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()


            if (epoch + 1) % 2 == 0:
                print("Saving checkpoint...")
                supervisor.saver.save(sess, (str(cfg.log_folder)+'/model_epoch_%04d_step_%02d') % (epoch, global_step))

        fd_train_acc.close()
        fd_loss.close()


def evaluation(model):
    loader = MnistLoader(image_filename=cfg.data_folder.joinpath("MNIST").joinpath("t10k-images.idx3-ubyte"),
                         label_filename=cfg.data_folder.joinpath("MNIST").joinpath("t10k-labels.idx1-ubyte"))
    num_te_batch = int(loader.size/cfg.batch_size)
    fd_test_acc = make_result_files()
    with model.graph.as_default() as g, tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(str(cfg.log_folder)))
        print('Model restored.')

        test_acc = 0
        for i in range(num_te_batch):
            labels, imgs = loader.batch(cfg.batch_size)
            labels = np.array(labels)
            zeros = np.zeros(len(labels), dtype=np.int32)
            imgs = np.array(imgs)
            acc = sess.run(model.accuracy, {model.X: imgs, model.labels: labels})
            predicted_label = sess.run(model.argmax_idx, {model.X: imgs, model.labels: zeros})
            test_acc += acc
            print("test "+str(i)+":")
            """
            for i in range(len(labels)):
                if labels[i] != predicted_label[i]:
                    print("at "+str(i)+" predicted: "+str(predicted_label[i])+", label:"+str(labels[i]))
                    plt.imshow(imgs[i].reshape([28,28]))
                    plt.show()
            """

        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        print('Test Finished.')

    fd_test_acc.close()

def main():
    model = CapsNet(is_training=cfg.is_training)

    if cfg.is_training:

        train(model)
    else:
        evaluation(model)
