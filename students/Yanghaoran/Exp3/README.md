# spatial transformer network.
- STN.pdf: The original paper that put forward the STN algorithm.
- STN_1_transformer.py: only have one transform matrix that acts on the input picture, not features.
- STN_2_transformer.py: have two transform matrix that acts on the input picture and the features on the first layer.
- STN_3_transformer.py: add another transformer matrix on the feature in the second layer.

> contrast 
1. as the increase of transform matrix, the test accuracy after the first epoch will decrease.
2. when add transform matrix into feature layers, the accuracy will decrease dramatically, for example, there exists such situation:
at  four epoch, the accuracy is 97%, but at five epoch, the accuracy will drop to 78%, and at next epoch, it is 97% again, so i think it is difficult to train when add matrix to features.
