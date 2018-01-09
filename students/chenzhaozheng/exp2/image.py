# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from skimage import io,color
from skimage import transform
from skimage.morphology import disk
import cv2
import sys
from predict import *
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters.rank as sfr

def get(i):
	if i<0 or i>=num or vis[i]==1:
		return 0
	vis[i] = 1
	return img[i//cols,i%cols]

def bfs(i):
	mxx,mxy = divmod(i,cols)
	mix = mxx
	miy = mxy
	q = []
	q.append(i)
	vis[i] = 1
	while len(q)>0:
		cur = q[0]
		q = q[1:]
		curx,cury = divmod(cur,cols)
		mxx = max(mxx,curx)
		mix = min(mix,curx)
		mxy = max(mxy,cury)
		miy = min(miy,cury)
		if get(cur-1)>con:
			q.append(cur-1)
		if get(cur+1)>con:
			q.append(cur+1)
		if get(cur-cols)>con:
			q.append(cur-cols)
		if get(cur+cols)>con:
			q.append(cur+cols)
		if get(cur-1-cols)>con:
			q.append(cur-1-cols)
		if get(cur+1-cols)>con:
			q.append(cur+1-cols)
		if get(cur-1+cols)>con:
			q.append(cur-1+cols)
		if get(cur+1+cols)>con:
			q.append(cur+1+cols)
	d = max(mxx-mix+1,mxy-miy+1)
	ans = np.zeros([d,d],dtype = np.float32)
	#print "mix = %d,miy = %d,mxx = %d,mxy = %d"%(mix,miy,mxx,mxy)
	print d
	for i in range(d):
		for j in range(d):
			if(d == mxx-mix+1):
				if j>-(mxy-miy)//2+d//2 and j<(mxy-miy)//2+d//2:
					ans[i,j] = img[i+mix,j+(mxy-miy)//2-d//2+miy]
				else:
					ans[i,j] = 0
			else:
				if i>-(mxx-mix)//2 + d//2 and i<(mxx-mix)//2+d//2:
					ans[i,j] = img[i+(mxx-mix)//2 - d//2+mix,j+miy]
				else:
					ans[i,j] = 0
	ans = cv2.copyMakeBorder(ans,d//4,d//4,d//4,d//4,cv2.BORDER_CONSTANT,value=0)
	return d,np.array(cv2.resize(ans,dsize=(28,28),interpolation=cv2.INTER_AREA).reshape(784),dtype = np.int32)


img_path = sys.path[0] + '/img4.jpg'
global img,num,vis,rows,cols,con
con = 40
img = cv2.imread(img_path,0)
#img = cv2.equalizeHist(img);
img = np.float32(img)
img = 255-img
rows = int(img.shape[0])
cols = int(img.shape[1])
print "rows=%d,cols=%d"%(rows,cols)

kernel = np.ones((3,3),np.float32)
kernel[0,0] = 0
kernel[0,1] = -1
kernel[0,2] = 0
kernel[1,0] = -1
kernel[1,1] = 5
kernel[1,2] = -1
kernel[2,0] = 0
kernel[2,1] = -1
kernel[2,2] = 0
#img =cv2.filter2D(img,-1,kernel)
plt.imshow(img)
plt.show()
for i in range(rows):
	for j in range(cols):
		if img[i,j]<160:
			img[i,j] = 0
kernel = np.ones((3,3),np.float32)/9
img =cv2.filter2D(img,-1,kernel)
#img = np.array(img,dtype = np.float32)
#img = cv2.equalizeHist(img);
for i in range(rows):
	for j in range(cols):
		if img[i,j]<128:
			img[i,j] = 0
img = np.array(img,dtype = np.int32)
plt.imshow(img)
plt.show()


num = rows*cols
out = []
vis = np.zeros([num],dtype = np.int32)
i = 0
while True:
	if get(i)>0:
		d,x = bfs(i)
		if(d>min(cols,rows)//8):
			out.append(x)
	i = i+cols
	if i>=num:
		i = (i+1)%num
	if i == num-1:
		break;

for i in range(len(out)):
	plt.imshow(out[i].reshape([28,28]))
	plt.show()

if len(out)>0:
	print "%d digits found"%(len(out))
	p = predict()
	p.predict(out)
else:
	print "no digit found"

