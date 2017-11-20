data=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\Aggregation.txt');
% data=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\mix.txt');
% data=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\flame.txt');
% data=load('C:\\Users\朱丽\\Downloads\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\R15.txt');
[m,n]=size(data);
figure();

minpts=10;
Eps=1.5;  %Aggregation  flame(效果很差)
% Eps=(1/3)*((prod(max(data)-min(data))*minpts*gamma(.5*(n-1)+1))/(m*sqrt(pi.^(n-1)))).^(1/(n-1));   %mix
% Eps=((prod(max(data)-min(data))*minpts*gamma(.5*(n-1)+1))/(m*sqrt(pi.^(n-1)))).^(1/(n-1));
%每列的极差连乘 * 邻域数目 * (1/2)n / (m^2*(npi))^(1/2n) 

x=[(1:m)' data]; %在数据集前面添加行号，形成新的数据矩阵
[m,n]=size(x); 
number=1; 
dealed=zeros(m,1);
types=zeros(1,m); 
class=zeros(1,m);
dis=p2p_dis(x(:,2:n-1));%不用标签，因此去掉最后一行

%颜色矩阵
A=[1,0,0;
    0,0,1;
    1,0,1;
    0,1,0;
    0,0,0;
    1,1,0;
    0,1,1;
    0.3,0.8,0.5;
    0.8,0.7,0.3;
    0.5,0,0.7;
    0.6,0.59,0.85;
    0.2,0.99,0.25;
    0.3,0.1,0.25;
    0.9,0.7,0.3;
    0.5,0.6,0.5;
    0.25,0.59,0.2;
    0.35,0.58,0.5;
    0.45,0.57,0.3;
    0.55,0.56,0.7;
    0.9,0.1,0.2;
    0.6,0.5,0.8;
    0.9,0.5,0.5;
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

for i=1:m
    if dealed(i)==0
        xTemp=x(i,:);
        D=dis(i,:);  %第i个点到其他所有点的距离
        ind=find(D<=Eps);
        
        %噪声
        if length(ind)==1
            types(i)=-1;
            class(i)=-1;
            dealed(i)=1;
        end
        
        %边缘
        if length(ind)>1 && length(ind)<=minpts
            types(i)=0;
            class(i)=0;
        end
       
        %中心点
        if length(ind)>minpts
            types(xTemp(1,1))=1;
            class(ind)=number;
            
            while ~isempty(ind)
                yTemp=x(ind(1),:);
                dealed(ind(1))=1;
                ind(1)=[];
                D=dis(yTemp(1,1),:);
                ind_1 = find(D<=Eps);
                
                if length(ind_1)>1
                    class(ind_1)=number;
                    if length(ind_1)>minpts
                        types(yTemp(1,1))=1;
                    else
                        types(yTemp(1,1))=0;
                    end
                    
                    for j=1:length(ind_1)
                        if dealed(ind_1(j))==0
                            dealed(ind_1(j))=1;
                            ind=[ind ind_1(j)];
                            class(ind_1(j))=number;
                        end
                    end
                end
            end
            number=number+1;
        end
    end
end

% ind_2=find(class==0);
% class(ind_2)=-1;
% types(ind_2)=-1;

for i=1:m
    dis(i,i)=100;
end


for i=1:m
    if class(i)==0||class(i)==-1
        [mi,indix]=min(dis(i,:));
        class(i)=class(indix);
    end
end

hold on
for i=1:m
    if class(i)==-1
        plot3(data(i,1),data(i,2),data(i,3),'o','color',A(1,:));
    else
        for k=1:30
            if class(i)==k
                plot3(data(i,1),data(i,2),data(i,3),'+','markersize',8,'color',A(k,:));
            end
        end
     end
end
hold off
xlabel('X');
ylabel('Y');
zlabel('Z');
title('DBSCAN');    