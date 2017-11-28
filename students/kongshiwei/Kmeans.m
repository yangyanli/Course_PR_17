[b] = xlsread('mix.xlsx',1,'A1：D1628');
x = b(:,1);
y = b(:,2);
%c = b(:,3);
%已经将数据集随机排序了所以直接取前23个作为样本
data = [x(1:23,1),y(1:23,1)];
%用于记录点到样本的距离
dist = zeros(1,23);
for k = 1:1000
%用来记录点被分到那个类中
c = zeros(1628,1);
sum = zeros(23,3);
for i = 1:1628
    for j = 1:23
        dist(1,j) = sqrt((x(i,1)-data(j,1))^2+(y(i,1)-data(j,2))^2);
    end
    [mi,index]=min(dist);
    c(i,1) = index;
    sum(c(i,1),1)=x(i)+sum(c(i,1),1);
    sum(c(i,1),2)=y(i)+sum(c(i,1),2);
    sum(c(i,1),3)=sum(c(i,1),3)+1;
end
%重新计算均值
for m = 1:23
    data(m,1) = sum(m,1)/sum(m,3);
    data(m,2) = sum(m,2)/sum(m,3);
end
end
%画图
for i = 1:1628
    rand('seed',c(i,1));
    color = rand(1,3);
    plot(x(i,1),y(i,1),'*','color',color);
    hold on;
end    