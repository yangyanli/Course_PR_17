[b] = xlsread('aggregate.xlsx',1,'A1：c788');
x = b(:,1);
y = b(:,2);
%c = b(:,3);
%已经将数据集随机排序了所以直接取前23个作为样本
data = [x(1:7,1),y(1:7,1)];
%用于记录点到样本的距离
dist = zeros(1,7);
for k = 1:300
%用来记录点被分到那个类中
c = zeros(788,1);
sum = zeros(7,3);
for i = 1:788
    for j = 1:7
        dist(1,j) = sqrt((x(i,1)-data(j,1))^2+(y(i,1)-data(j,2))^2);
    end
    [mi,index]=min(dist);
    c(i,1) = index;
    sum(c(i,1),1)=x(i)+sum(c(i,1),1);
    sum(c(i,1),2)=y(i)+sum(c(i,1),2);
    sum(c(i,1),3)=sum(c(i,1),3)+1;
end
%重新计算均值
for m = 1:7
    data(m,1) = sum(m,1)/sum(m,3);
    data(m,2) = sum(m,2)/sum(m,3);
end
end
%画图
for i = 1:788
    rand('seed',c(i,1));
    color = rand(1,3);
    plot(x(i,1),y(i,1),'*','color',color);
    hold on;
end    