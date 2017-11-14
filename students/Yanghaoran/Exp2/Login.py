# -*- coding:utf-8 -*-
import importlib
import sys
import numpy as np
importlib.reload(sys)
#sys.setdefaultencoding('utf8')

import urllib
#import urllib2
import http.cookiejar
import re
#import cookielib
import sys
import os
import string
import ProcessImage
class YJSSpider:
    # 模拟登陆研究生教务系统
    def __init__(self):
        self.baseURL = "http://58.194.172.34/reader/login.php"
        self.enable = True
        self.charaterset = "gb2312"
        string = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:56.0) Gecko/20100101 Firefox/56.0"
        self.headers = {'User-Agent' : string}
        self.cookie = http.cookiejar.CookieJar()
        self.hander = urllib.request.HTTPCookieProcessor(self.cookie)
        self.opener = urllib.request.build_opener(self.hander)

    # 验证码处理
    def getCheckCode(self,net):
        # 验证码连接
        checkcode_url = "http://58.194.172.34/reader/yz.php"
        request = urllib.request.Request(checkcode_url, headers=self.headers)
        picture = self.opener.open(request).read()
        # 将验证码写入本地
        local = open("checkcode.jpg", "wb")
        local.write(picture)
        local.close()
        # 调用系统默认的图片查看程序查看图片
        os.system("checkcode.jpg")
        img_list=ProcessImage.MainProcess()
        txt_check=""
        for img in img_list:
            #img.show()
            img=np.reshape(img, (784, 1))
            img2=np.zeros((784,1))
            for i in range(784):
                img2[i][0]=float((255-img[i][0])/255.0)
            #img=float(img/[[255.0] for i in range(784)])
            #print(img[0].shape)
            num=net.GetNumber(img2)
            print(num)
            txt_check=txt_check+str(num)
        print(txt_check)

        #txt_check = input(str("请输入验证码").encode(self.charaterset))
        return txt_check

    # 模拟登陆
    def login(self, userid, userpwd, txt_check):
        #获取验证码
        #txt_check = self.getCheckCode()
        postData = {"code":txt_check, "number":userid, "passwd":userpwd,"returnUrl":""}
        data = urllib.parse.urlencode(postData).encode(encoding="UTF-8")

        request_url = "http://58.194.172.34/reader/redr_verify.php"
        request_new = urllib.request.Request(request_url, headers=self.headers)
        response = self.opener.open(request_new, data)
        print(response.read().decode("utf-8"))
        '''
        if response.status_code == 200:
            print('恭喜您登陆成功！')
        else:
            print(response.text)
            print('登录失败，请重试！')
        '''
