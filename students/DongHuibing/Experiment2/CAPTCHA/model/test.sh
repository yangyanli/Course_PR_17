#!/usr/bin/env sh
cd /home/gdymind/Desktop/caffe/build/tools/
./caffe.bin test \
-model /home/gdymind/Desktop/PatternExp2/model/train.prototxt \
-weights /home/gdymind/Desktop/PatternExp2//snapshot/snapshot_iter_48000.caffemodel \
-iterations 20
