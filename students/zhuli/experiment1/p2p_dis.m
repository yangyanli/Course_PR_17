function [dis] = p2p_dis(x)
    [m,n]=size(x);
    dis=zeros(m,m);

for i=1:m
    for j=i:m
        %两点间的欧式距离
        tmp=0;
        for k=1:n
            tmp=tmp+(x(i,k)-x(j,k)).^2;
        end
        dis(i,j)=sqrt(tmp);
        dis(j,i)=dis(i,j);
    end
end

