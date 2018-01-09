% data=load('F:\Git\Course_PR_17\experiment1\data\\synthetic_data\\Aggregation.txt');
data=load('F:\Git\Course_PR_17\experiment1\data\\synthetic_data\\mix.txt');
% data=load('F:\Git\Course_PR_17\experiment1\data\\synthetic_data\\flame.txt');
% data=load('F:\Git\Course_PR_17\experiment1\data\\synthetic_data\\R15.txt');
data = data(:,1:end-1);
[m,n]=size(data);
figure();

% 聚类簇数，迭代次数，指数
cluster = 12;
iter = 50;
index = 2;

% 隶属度
U = rand(cluster,m);
col_sum = sum(U);
U = U./col_sum(ones(cluster,1),:);

% 聚类
for i = 1:iter
    % c
    for j = 1:cluster
        u_ij = U(j,:).^index;
        sum_ij = sum(u_ij);
        sum_ld = u_ij./sum_ij;
        c(j,:) = u_ij*data./sum_ij;
    end
    % J
    temp_1 = zeros(cluster,m);
    for j = 1:cluster
        for k = i:m
            temp_1(j,k) = U(j,k)^index * (norm(data(k,:) - c(j,:)))^2;
        end
    end
    J(i) = sum(sum(temp_1));
    
    % U
    for j = 1:cluster
        for k = 1:m
            sum_1 = 0;
            for j1 = 1:cluster
                temp = (norm(data(k,:)-c(j,:))/norm(data(k,:)-c(j1,:))).^(2/(index-1));
                sum_1 = sum_1 + temp;
            end
            U(j,k) = 1./sum_1;
        end
    end
end

subplot(1,2,1),plot(J);
[~,label] = max(U);
subplot(1,2,2);
gscatter(data(:,1),data(:,2),label)