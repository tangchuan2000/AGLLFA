clear;
X = [2 5 6 9 1 9; 7  4 6 8 3 7; 4 7 8 1 3 4];
k = 4;

 [~, n] = size(X);
% D = L2_distance_1(X, X);
% [~, idx] = sort(D, 2); % sort each row

% S = zeros(n);
% for i = 1:n
%     id = idx(i,2:k+2);
%     di = D(i, id);
%     S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
% end;

D = zeros(n);
for i=1:n
    for j = 1:n
        D(i,j) = norm(X(:,i) - X(:,j))^2;
    end
end
[~, idx] = sort(D, 2); % sort each row

S2 = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);
    di = D(i, id);
    S2(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;
tc =1 ;


X = [2 5 6 ; 
    7 4 3;
    8 3 7]
A = X';
tc = max(X, X');


n = 5;
I = diag(ones(1,n));
One = ones(n,1);
H = I - One * One' /n
A = H * One;

Y = [1 1 1;
    2 1 3;
    2 5 7;
     2 2 3];
% F = Y * (Y' * Y)^(-0.5);
% %G = (diag(diag(F * F')))^(-0.5) * F;
% G = (diag(F * F'))^(-0.5) * F;
% %G1 = inv((diag(diag(F * F'))^0.5) * F;
% H = F * F';
% H2 = F' * F;
% 
% tc =1 ;


D = diag([2 1 -1 -2 ]);
B = Y' * D * Y;


%save('t.mat', "Y" );  
% X = [1, 2, 3;
%     4, 5, 6;
%     7, 8, 9]
v = [1, 2, 3]
B = repmat(v, 3, 1) %行复制3次，列不复制
