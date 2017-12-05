from PIL import Image
from numpy import *

def GetImage(filelist):
	width=28
	height=28
	value=zeros([1,width,height,1])
	value[0,0,0,0]=-1
	for filename in filelist:
		img=array(Image.open(filename).convert("L"))
		width,height=shape(img);
		index=0
		tmp_value=zeros([1,width,height,1])
		for i in range(width):
			for j in range(height):	
				tmp_value[0,i,j,0]=img[i,j]
				index+=1
		if(value[0,0,0,0]==-1):
			value=tmp_value
		else:
			value=concatenate((value,tmp_value))
	return array(value)
