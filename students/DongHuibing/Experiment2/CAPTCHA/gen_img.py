import string
import os
import random
import numpy as np
import json
from captcha.image import ImageCaptcha
from operator import itemgetter
from PIL import Image
import h5py

image = ImageCaptcha(width=100, height=40, fonts=['Arial.ttf'])
work_dir = "/home/gdymind/Desktop/PatternExp2/pre_data/"
data_dir = "/home/gdymind/Desktop/PatternExp2/data/"

def get_label(captcha):
    label = []
    for ch in captcha:
        x = char2num_map[ch]
        label.append(x)
    return label

def gen_img(name, captcha_len, sample_num):
    #create image folder
    img_dir = work_dir + name + '_image/'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
        print "create folder: " + img_dir

    label_map = {}
    
    #create image files
    for i in range(sample_num):
        cur_text =  ''
        for j in range(captcha_len):
            cur_text += random.choice(charset)
        img_name = str(i) + '.png'
        label_map[img_name] = get_label(cur_text)
        image.generate(cur_text)
        image.write(cur_text, work_dir + name + '_image/' + img_name)
        #if i >= 2000 and i % 2000 == 0:
            #print str(i) + " " + name + " images generated"

    #generate json files
    json.dump(label_map, open(work_dir+ name + ".json", "w"))
    #print  name + ".json generated"

def json2txt(s, out):
    f = open(s)
    json_content = json.load(f)
    l = list(json_content.iteritems())
    f.close()
    with open(out, 'w+') as L:
        for t in l:
            L.write(t[0] + ' ')
            for i in t[1]:
                L.write(str(i) + ' ')
            L.write('\n')
    print out + " generated"  
    
def gen_h5(name, captcha_len):
    lines = open(work_dir + name + '.txt', 'r').readlines()
    img_num = len(lines)

    Data = np.zeros((img_num, 1, 100, 40), dtype='f4')
    Label = np.zeros((img_num, captcha_len), dtype='f4')

    for i, line in enumerate(lines):
        sp= line.split(' ')
        sp.pop() #remove '\n'
        img_dir = work_dir + name + '_image/'
        img = Image.open(img_dir + sp[0]).convert("L")#rgb to gray
	img = img.transpose(Image.ROTATE_90)#.transpose(Image.FLIP_LEFT_RIGHT)
	#img.save(img_dir + sp[0])
	img_a = np.array(img, dtype='f4')
        Data[i][0] = img_a / 255.0
        Label[i] = np.array(sp[1:], dtype='f4')

    with h5py.File(data_dir+name + '.h5', 'w') as H:
        H.create_dataset('data', data=Data)
        H.create_dataset('label', data=Label)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print "create folder: " + img_dir
    if name[0] == 't':
        open(data_dir+'train_h5list.txt', 'a+').write(data_dir+name + '.h5\n')
    else:
        open(data_dir+'val_h5list.txt', 'a+').write(data_dir+name + '.h5\n')
    print name + ".h5 generated."

    
if __name__ == "__main__":
    captcha_len = 4
    train_num = 5000
    val_num = 10
    
    charset = [a for a in string.ascii_letters] + [d for d in string.digits]
    charset.remove('1')
    charset.remove('l')
    charset.remove('I')
    char2num_map = dict(zip(charset, range(len(charset))))

    for i in range(100):
        gen_img('train' + str(i), captcha_len, train_num)
        gen_img('val' + str(i), captcha_len, val_num)

        json2txt(work_dir + "train"+ str(i)+".json", work_dir + "train"+ str(i)+".txt")
        json2txt(work_dir + "val"+ str(i)+".json",  work_dir + "val"+ str(i)+".txt")

        gen_h5("train"+ str(i), captcha_len)
        gen_h5("val"+ str(i), captcha_len)
