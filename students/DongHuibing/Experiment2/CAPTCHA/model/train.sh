#!/usr/bin/env sh
cd /home/gdymind/Desktop/caffe/build/tools/
./caffe train -solver=/home/gdymind/Desktop/PatternExp2/model/solver.prototxt \
-snapshot=/home/gdymind/Desktop/PatternExp2//snapshot/snapshot_iter_43880.solverstate
