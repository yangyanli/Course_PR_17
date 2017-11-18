clear;
clc;
a = textread('E:\обть\Course_PR_17-master\experiment1\data\synthetic_data\kflame.txt');
x = a(:,1);
y = a(:,2);
lable = a(:,3);
scatter(x,y,10,lable','filled');