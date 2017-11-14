import Login
import network
net=network.load("model")

t=Login.YJSSpider()
txt_check=t.getCheckCode(net)
userid="input your user id"
userpwd="input you user pwd"
t.login(userid,userpwd,txt_check)
