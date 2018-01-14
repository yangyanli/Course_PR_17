import os
import tensorflow as tf
# import numpy as np
#
# def next_batch(num,data,label):
#     '''
#     Return a total of `num` random samples and labels.
#     '''
#     idx = np.arange(0, len(data))
#     np.random.shuffle(idx)
#     idx = idx[:num]
#     data_shuffle = [data[i] for i in idx]
#     labels_shuffle = [label[i] for i in idx]
#
#     return np.asarray(data_shuffle), np.asarray(labels_shuffle)
#
#
# data = [1,2,3,4,5,6,7,8]
# label = [1,2,3,4,5,6,7,8]
#
# for i in range(1000000):
#
#     print(next_batch(3,data,label))

from PIL import ImageFilter
from PIL import Image

im = Image.open('genpic/afetr.png')

# im = im.convert("1")

im = im.convert("L")
# im = im.filter(ImageFilter.MaxFilter(1))
# im = im.filter(ImageFilter.MinFilter(3)).show()

# data = im.getdata()
# w,h = im.size
# #im.show()
# black_point = 0
# for x in range(1,w-1):
#     for y in range(1,h-1):
#         mid_pixel = data[w*y+x] #中央像素点像素值
#         if mid_pixel == 0: #找出上下左右四个方向像素点像素值
#             top_pixel = data[w*(y-1)+x]
#             left_pixel = data[w*y+(x-1)]
#             down_pixel = data[w*(y+1)+x]
#             right_pixel = data[w*y+(x+1)]
#
#             #判断上下左右的黑色像素点总个数
#             if top_pixel == 0:
#                 black_point += 1
#             if left_pixel == 0:
#                 black_point += 1
#             if down_pixel == 0:
#                 black_point += 1
#             if right_pixel == 0:
#                 black_point += 1
#             if black_point >= 3:
#                 im.putpixel((x,y),0)
#             #print black_point
#             black_point = 0
# im.show()
#
#
#
# def Clear_Point(im):
#     for j in range(1,(im.size[1]-1)):
#         for i in range(1,(im.size[0]-1)):
#             if im.getpixel((i,j))==0 and im.getpixel(((i-1),(j-1)))==255  and im.getpixel((i,(j-1)))==255  and im.getpixel(((i +1),(j-1)))==255  and im.getpixel(((i-1),j))==255  and im.getpixel(((i +1),j))==255  and im.getpixel(((i-1),(j+ 1)))==255  and im.getpixel((i,(j +1)))==255  and im.getpixel(((i +1),(j +1)))==255:
#                 im.putpixel([i,j],255)
#     return im
#
#
#
import numpy as np

pred = np.array([[[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]],
            [[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]]])
sess = tf.InteractiveSession()

print(tf.argmax(pred,axis = 1).eval(session=sess))





