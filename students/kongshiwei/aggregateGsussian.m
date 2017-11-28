[b] = xlsread('aggregate.xlsx',1,'A1：C788');
%归一化
x = b(:,1)/max(b(:,1));
y = b(:,2)/max(b(:,2));
c = zeros(788,1);
axis=[x,y];
%已经将数据集随机排序了所以直接取前7个作为样本
data = [x(1:7,1),y(1:7,1)];
p = (zeros(7,1)+1)/7;
variance = cell(1,7);
for i = 1:7
    variance{i} = [0.1 0;0 0.1];
end
r = zeros(788,7);
for q =1:40
for i = 1:788
    sumR = 0;
    for j = 1:7
        sumR = sumR+1/sqrt(det(variance{j}))*exp(-0.5*(axis(i,:)-data(j,:))/variance{j}*((axis(i,:)-data(j,:)).'))*p(j,1);
    end
    for k = 1:7
        r(i,k) = 1/sqrt(det(variance{k}))*exp(-0.5*(axis(i,:)-data(k,:))/variance{k}*((axis(i,:)-data(k,:)).'))*p(j,1)/sumR;
    end
end
for m = 1:7
    tempD = [0,0];
    %更新均值
    for n =1:788
       tempD = tempD + r(n,m)*axis(n,:); 
    end
    data(m,:) = tempD/sum(r(:,m));
    %更新方差
    tempV = zeros(2,2);
    for n = 1:788
        tempV = tempV + r(n,m)*((axis(n,:)-data(m,:)).'*(axis(n,:)-data(m,:)));
    end
    variance{m} = tempV/sum(r(:,m));
    %更新系数
    p(m) = sum(r(:,m))/788;
end
end
for i = 1:788
    [mi,index] = max(r(i,:));
    c(i)=index;
end
%画图
for i = 1:788
    rand('seed',c(i,1));
    color = rand(1,3);
    plot(x(i,1),y(i,1),'*','color',color);
    hold on;
end   