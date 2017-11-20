dim = 3; %维数
k = input('k=');  
% PM=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\Aggregation.txt');
PM=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\flame.txt');
% PM=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\mix.txt');
% PM=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\R15.txt');
% PM=load('D:\\Matlab\\testSet.txt');
[N,M] = size(PM);
figure();%创建窗口

CC = zeros(k,dim); % 聚类中心矩阵
D = zeros(N,k); % D(i,j)是样本i和聚类中心j的距离

%cell是元包数组，类似结构体，每个元素可以不一样
C = cell(1,k); % 聚类矩阵初始化，对应聚类包含的样本
for i = 1:k-1
    C{i} = [i];
end
C{k} = k:N;
B = 1:N; %上次迭代中，样本属于哪一聚类，设初值为1
B(k:N) = k; %k到N个样本属于第k类

%初始化聚类中心矩阵,随机取
for i = 1:k
        n=randi(size(PM,1));
        CC(i,:)=PM(n,:);
end

%颜色矩阵
A=[1,0,0;
    0,1,0;
    0,0,1;
    1,0,1;
    0,0,0;
    1,1,0;
    0,1,1;
    0.3,0.8,0.5;
    0.4,0.7,0.3;
    0.5,0.6,0.7;
    0.6,0.59,0.85;
    0.2,0.99,0.25;
    0.3,0.89,0.55;
    0.4,0.79,0.35;
    0.5,0.69,0.75;
    0.25,0.59,0.2;
    0.35,0.58,0.5;
    0.45,0.57,0.3;
    0.55,0.56,0.7;
    0.9,0.1,0.2;
    0.6,0.5,0.8;
    0.6,0.5,0.8;
    0.7,0.4,0.2;
    0.8,0.3,0.2;
    0.9,0.2,0.2;
    1,1,0.8;
    1,0.2,0.5;
    0.9,0.1,0.2;
    0.5,0.15,0.8;
    0.51,0.2,0.5;
    0.15,0.1,0.22;
    0.25,0.66,0.12;
    0.35,0.25,0.29;
    0.45,0.85,0.25];

while true   
    change = 0;
    %样本i到k个聚类中心的距离
    for i = 1:N
        for j = 1:k
              D(i,j) = sqrt((PM(i,1) - CC(j,1))^2 + (PM(i,2) - CC(j,2))^2);
        end
        t = find( D(i,:) == min(D(i,:)) ); % i属于第t类
        if B(i) ~= t % 上次迭代i不属于第t类
            change = 1;
            % 将i从第B(i)类中去掉
            t1 = C{B(i)};
            t2 = find( t1==i );            
            t1(t2) = t1(1);
            t1 = t1(2:length(t1)); 
            C{B(i)} = t1;
            C{t} = [C{t},i]; % 将i加入第t类
            B(i) = t;
        end        
    end
    if change == 0 
        break;
    end

    % 重新计算CC
    for i = 1:k
        CC(i,:) = 0;
        iclu = C{i};
        for j = 1:length(iclu)
            CC(i,:) = PM( iclu(j),: )+CC(i,:);
        end
        CC(i,:) = CC(i,:)/length(iclu);
    end
end

plot(CC(:,1),CC(:,2),'o') 
hold on

for i=1:N
    for x=1:k
        if(B(1,i)==x)
            plot3(PM(i,1),PM(i,2),PM(i,3),'.','markersize',15,'color',A(x,:));
             hold on
        end
    end
end
xlabel('X');
ylabel('Y');
zlabel('Z');
title('k-means');    