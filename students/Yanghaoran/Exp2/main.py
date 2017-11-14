import Login
import network
net=network.load("model")

t=Login.YJSSpider()
txt_check=t.getCheckCode(net)
userid="259150"
userpwd="201502"
t.login(userid,userpwd,txt_check)
