
from captcha.image import ImageCaptcha
import string
import PIL
from PIL import Image
import numpy as np


#
# # data = image.generate('12afeaf34')
# # assert isinstance(data, BytesIO)
#
#
# im = Image.open("genpic/5f5e.png")
# # im.show()
# im2 = im.convert("1")
#
# a = np.reshape(np.array(im2),[9600])
#
# a = [int(i) for i in a ]
#
# a = np.array(a)
#
# print(a)
# im2.save('genpic/kk55565665k.png')
import random

def random_string(size, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def trans_num_index(num):
    return int(num)-1

def trans_letter_index(letter):
    return ord(letter)-88

def generate(model,size,num,width,height):
    print(model)
    image = ImageCaptcha(width=width,height=height)

    labels = np.zeros((num,size*10))

    label_text = open('genpic/'+model+'/label.txt','w',encoding='utf8')

    for i in range(num):
        mystring = random_string(size)
        for j in range(len(mystring)):

            labels[i][j*10+int(mystring[j])] += 1


        if i%1000 == 0:
            print('finsh'+str(i))
        label_text.write(str(i)+' '+mystring+'\n')

        image.write(mystring,'genpic/'+model+'/'+str(i)+'.png')

    np.save('genpic/'+model+'_label'+'.npy',labels)







if __name__ == '__main__':
    generate('test_num', size = 4,num = 2*10**4,width=160,height=60)
    generate('train_num',size = 4,num = 2*10**4,width=160,height=60)

    # image = ImageCaptcha(width=60,height=60)
    # image.write('f',"genpic/ff.png")
    # im = Image.open("genpic/5f5e.png")

    # print(im.size)