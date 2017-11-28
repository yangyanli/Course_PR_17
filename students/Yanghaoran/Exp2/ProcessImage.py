from PIL import Image
import numpy as np

def Binear(filename):
    img = Image.open(filename)

    Lim = img.convert('L')
    return Lim
def Resize(img):
    img2=img.resize((img.size[0]*4, img.size[1]*4), Image.ANTIALIAS)
    return img2

def binarizing(img,threshold): #input: gray image
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < 120:
                pixdata[x, y] = 0
            else:
               if pixdata[x,y]>200:
                   pixdata[x,y]=255
    '''
    for y in range(h):
        for x in range(w):
            print(pixdata[x,y]," ",end='')
        print()
    '''
    return img

def depoint(img):   #input: gray image
    pixdata = img.load()
    w,h = img.size

    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > 250:
                count = count + 1
            if pixdata[x,y+1] > 250:
                count = count + 1
            if pixdata[x-1,y] > 250:
                count = count + 1
            if pixdata[x+1,y] > 250:
                count = count + 1
            if count > 2:
                pixdata[x,y] = 255
    return img

def Split(img):
    width,height=img.size
    visited=np.zeros((width,height))
    k=1
    SegCount = [0 for i in range(28 * 28)]
    Record=[]
    pix = img.load()
    for i in range(width):
        for j in range(height):
            if pix[i,j]<200 and visited[i][j]==0:
                maxx=i
                maxy=j
                minx=i
                miny=j
                visited[i][j]=k
                SegCount[k]+=1
                minx, miny, maxx, maxy=FillFlood(pix,i,j,visited,k,width,height,SegCount,minx,miny,maxx,maxy)
                Record.append([minx,miny,maxx,maxy])
                k+=1
    return visited,SegCount,k,Record
def FillFlood(pix,i,j,visited,k,width,height,SegCount,minx,miny,maxx,maxy):
    if width>i+1 and visited[i+1][j]==0 and pix[i+1,j]<200:
        SegCount[k] += 1
        visited[i+1][j]=k
        maxx=max(maxx,i+1)

        minx, miny, maxx, maxy=FillFlood(pix,i+1,j,visited,k,width,height,SegCount,minx,miny,maxx,maxy)
    if i-1>=0 and visited[i-1][j]==0 and pix[i-1,j]<200:
        SegCount[k] += 1
        visited[i-1][j]=k
        minx=min(minx,i-1)

        minx, miny, maxx, maxy=FillFlood(pix,i-1,j,visited,k,width,height,SegCount,minx,miny,maxx,maxy)

    if j-1>=0 and visited[i][j-1]==0 and pix[i,j-1]<200:
        SegCount[k] += 1
        visited[i][j-1]=k
        miny=min(miny,j-1)

        minx, miny, maxx, maxy=FillFlood(pix,i,j-1,visited,k,width,height,SegCount,minx,miny,maxx,maxy)

    if height>j+1 and visited[i][j+1]==0 and pix[i,j+1]<200:
        SegCount[k] += 1
        visited[i][j+1]=k
        maxy=max(maxy,j+1)

        minx, miny, maxx, maxy=FillFlood(pix,i,j+1,visited,k,width,height,SegCount,minx,miny,maxx,maxy)
    return minx,miny,maxx,maxy
def Processing(img,visited,segcount,k,Record,Min=80):
    pix=img.load()
    width=img.size[0]
    height=img.size[1]
    pp=0
    for p in range(1,k):
        if(segcount[p]<Min):
            del Record[p-1-pp]
            pp+=1
            for i in range(width):
                for j in range(height):
                    if(visited[i][j]==p):
                        visited[i][j]=0
                        pix[i,j]=255
    return img,visited,Record

def Split_Image(img,Record):
    img_list=[]
    for i in range(len(Record)):
        img2 = img.crop((Record[i][0], Record[i][1], Record[i][2], Record[i][3]))
        img2 = img2.resize((20, 20), Image.ANTIALIAS)
        img_list.append(img2)
    img_list2=[]
    for i in range(len(img_list)):
        img2=img_list[i]
        img3=Image.new(mode='L',size=(28,28),color=255)
        img3.paste(img2,(4,4))
        img_list2.append(img3)
    return img_list2

def MainProcess():
    img = Binear("checkcode.jpg")
    img2 = Resize(img)
    img3 = binarizing(img2, 100)
    #img4 = depoint(img2)
    #img3.show()
    visited, SegCount, k, Record = Split(img3)
    img4, visited, Record = Processing(img3, visited, SegCount, k, Record)
    #img4.show()
    img_list = Split_Image(img4, Record)

    return img_list







