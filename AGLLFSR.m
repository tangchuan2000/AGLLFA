function [acc, nmi, ARI, Fscore, Pu, Precision, Recall] = AGLLFSR(X, bNorm, knn, gnd, maxIter, lamda)   
    viewNum = numel(X);
    sampleNum = size(X{1},2);
    cluNum = length(unique(gnd));

    %% Normalization
%     if (bNorm == 1)
%         for v = 1: viewNum
%             X{1,v} = NormalizeFea(X{1, v}, 0);
%         end
%     end
    %tic;
%     if bNorm == 1
%         parfor i = 1:viewNum
%             for  j = 1:sampleNum
%                 normItem = std(X{i}(:,j));
%                 if (0 == normItem)
%                     normItem = eps;
%                 end
%                 X{i}(:,j) = (X{i}(:,j)-mean(X{i}(:,j)))/(normItem);
%             end
%         end
%     end
%     fprintf('Normalization1 parfor :%.4f sampleNum %d\n', toc, sampleNum);
    
     %tic;     
     if bNorm == 1
        for i = 1:viewNum
            normRowVector = full(std(X{i}));
            feaNum = size(X{i}, 1);
            normRowVector(normRowVector == 0) = eps;
            meanRowVector = mean(X{i});
            normMat = repmat(normRowVector, feaNum, 1); 
            X{i} = (X{i} - meanRowVector) ./ normMat;    
        end
    end
    %fprintf('Normalization2 :%.4f sampleNum %d\n', toc, sampleNum);
    
    
    %% initialize X'*X  used many times
     % tic; 
    XTX = cell(1,viewNum); % X'*X
    for v = 1:viewNum
        XTX{v} = zeros(sampleNum, sampleNum);
        %XTX{v} = X{v}' * X{v};
        XTX{v} = full(X{v}' * X{v});    
    end
    % fprintf('initialize XX:%.4f sampleNum %d\n', toc, sampleNum);
    
    %% 
     %tic;
    B = cell(1,viewNum); %inverse    
    for v = 1:viewNum
        B{v} = zeros(sampleNum, sampleNum);
        B{v} = inv(XTX{v} + lamda(1) * eye(size(X{v},2)));
    end
     %fprintf('initialize B :%.4f sampleNum %d\n', toc, sampleNum);
    
    
    %% initialize S
    % tic;
    S = cell(1,viewNum);
    for v = 1:viewNum
        S{v} = zeros(sampleNum, sampleNum);
        S{v} = B{v} * X{v}' * X{v} ;
        S{v}(S{v} < 0) = 0;
        S{v} = S{v} - diag(diag(S{v}));
    end
     %fprintf('initialize S_v :%.4f sampleNum %d\n', toc, sampleNum);

    %% initialize A D L 
    A = cell(1,viewNum);
    D = cell(1,viewNum);     
    L = cell(1,viewNum);
    %tic;
    for v = 1:viewNum    
        A{v} = zeros(sampleNum, sampleNum);
        L{v} = zeros(sampleNum, sampleNum);
        D{v} = zeros(sampleNum, sampleNum);
        A{v} = (S{v} +  S{v}') / 2;  
        vectorDNN = sum(A{v},2);
        D{v} = diag(vectorDNN);      
        L{v} = D{v} - A{v};
    end
	%fprintf('initialize A D L:%.4f sampleNum %d\n', toc, sampleNum);
    
  
  
     %% initialize F
    F = cell(1,viewNum);
     %tic;
     F1 =  cell(1,viewNum);
     F2 =  cell(1,viewNum);
    for v = 1:viewNum
        F{v} = zeros(sampleNum, cluNum);
       [F{v}, ~, ~]=eig1(2 * L{v}, cluNum, 0, 0);
       F1{v} = sum(F{v});
       F2{v} = sum(sum(F{v}));
    end  
     %fprintf('initialize F :%.4f sampleNum %d\n', toc, sampleNum);
    
    
    %% initialize beta
    beta = ones(1, viewNum).*(1/sqrt(viewNum));

    %% initialize WP
    WP = cell(1,viewNum);   
    for v = 1:viewNum
       WP{v} = eye(cluNum);
    end
    
    
        
    %%
    error = zeros(maxIter,1);
    fz = 1e-6 * lamda(2);



    for iter = 1:maxIter
        tic;        
        %% update S_v    
        lamda2 = lamda(2);
        lamda3 = lamda(3);
        %loadlibrary('UpdateSvForLFSR','lfsr.h');
        for v = 1:viewNum     
           % tic;   
   
% % % % % % % % % % % % % % % % %% % % % % % mexFunction % % % % % % % % % % %% % % % % % % % % % % % % % % % %
%%%%% complie  mex updateSv.cpp -output updateSv -IE:\eigen-3.3.9
        %tic        
         h = updateSv(F{v}, B{v}, X{v}, lamda2);  
         S{v} = B{v} * (XTX{v} - lamda2 / 2 * h');
         S{v}(S{v} < 0) = 0;
         S{v} = S{v} - diag(diag(S{v}));
         %fprintf('update S_v time by mexFunction:%.4f sampleNum %d\n', toc, sampleNum);
% % % % % % % % % % % % % % % % %% % % % % % % % % % % % % % % % %% % % % % % % % % % % % % % % % %
        end    
        %fprintf('update S_v time by mexFunction :%.4f sampleNum %d\n', toc, sampleNum);   
 
        %% update Y
        %tic
        sumFW = zeros(sampleNum, cluNum);
        for v = 1:viewNum
            sumFW = sumFW + beta(v) * F{v} * WP{v};
        end
        [U,~,V] = svd(sumFW, 'econ');
        Y = U * V';
        %fprintf('update  Y:%.4f \n', toc);
        %tc1 = Y' * Y; % Y' Y constrained to I,here test it
        
        %tic
       %% update WP        
        for v = 1:viewNum
            FY = beta(v) * F{v}' * Y;
            [U,~,V] = svd(FY);
            WP{v} = U * V';
            %tc1 = WP{v}' * WP{v}; % WP{v}' WP{v} constrained to I,here test it            
        end
        %fprintf('update WP:%.4f \n', toc);
        
        
        % tic
       %% update F
        for v = 1:viewNum            
            A{v} = (S{v} +  S{v}') / 2; 
%             for n = 1:sampleNum
%                 D{v}(n, n) = sum(A{v}(n, :));
%             end
            vectorDNN = sum(A{v},2); 
            D{v} = diag(vectorDNN);      
            L{v} = D{v} - A{v};
        end
        % fprintf('update  F1:%.4f \n', toc);
%         tic
        for v = 1:viewNum            
            b = lamda3 / (4 * lamda2) * beta(v) * Y * WP{v}';
            last_F_cost = 0;
            cost = 1e-4;
            times = 0;
            
            Lvector = diag(L{v});
            maxElement = max(Lvector);
            alpha = ceil(maxElement);
            LP = alpha * eye(sampleNum) - L{v}; % LP is a positive matrix, GPI by << A generalized power iteration method for solving
                                         %    quadratic problem on the Stiefel manifold >> page 3 to
                                         %    slove Eq (2)
            while (cost - last_F_cost > 1e-5 && times < 20)
                last_F_cost = cost;
                [FF, cost] = GPI(LP, F{v}, b);  
                F{v} = FF;
                times = times + 1;
            end
            %tc1 = F{v}' * F{v}; % F' * F constrained to I,here test it
        end        
%         fprintf('update  F2:%.4f \n', toc);
        
        %% update beta  
        sumTmp = 0;
        for v = 1:viewNum
           tmp = trace(Y' * F{v} * WP{v});   
           tmp = tmp^2;
           sumTmp = sumTmp + tmp;
        end
        sumTmp = sqrt(sumTmp);
        for v = 1:viewNum
            tmp = trace(Y' * F{v} * WP{v});
            beta(v) = tmp / sumTmp;
        end

         

         
      
         %% get result
         [acc, nmi, ARI, Fscore, Pu, Precision, Recall] = getResultByKmeans(Y, gnd, cluNum);  

        %% update lost      
                  
        if(maxIter > 1) 
            error(iter) = cost_function(X, S, F, Y, lamda, WP, beta, L, cluNum); 
            if (iter > 4 ) 
                decreaseRatio = (error(iter - 1) - error(iter)) / error(iter);
                if (decreaseRatio < fz) 
                    %fprintf(' decreaseRatio:%.10f', decreaseRatio);
                    break;
                end
                %fprintf('  iter:%d  decreaseRatio:%.10f', iter, decreaseRatio);
            end        
        end
        
    end % for 
    clear F Y X WP beta A L D S error XTX B; 
    
end

function [F, cost] = GPI(SumA, F, b)
    %for i = 1:20
        M = 2 * SumA * F + 2 * b;
        [U,~,V] = svd(M,'econ');
        F = U * V';
        cost = F_cost(SumA, F, b);
        %fprintf('F_cost:%.6f\n', cost);
    %end

end

function err = F_cost(SumA, F, b)
    err = trace(F' * SumA * F + 2 * F' *b);
end

function err = cost_function(X, S, F, Y, lamda, WP, beta, L, cluNum)
	sampleNum = size(X{1},2);
    sum0 = 0;
    sum1 = 0;
    viewNum = numel(X);
    for v = 1:viewNum
        nor = norm(X{v} - X{v} * S{v}, 'fro');
        sum0 = sum0  + nor^2 ;            
    end
    
    for v = 1:viewNum        
           sum1 = sum1 + norm(S{v},  'fro')^2;        
    end
    
    sum2 = 0;
    for v = 1:viewNum
        sum2 = sum2 + trace(F{v}' * L{v} * F{v});
    end
    
    
    tmp = zeros(sampleNum, cluNum);
    for v = 1:viewNum
        tmp = tmp + beta(v) * F{v} * WP{v};
    end        
    sum3 = trace(Y' * tmp);
     
    err = sum0 + lamda(1) * sum1 + 2 * lamda(2) * sum2 - lamda(3) * sum3;
     %str = sprintf(' [sum0 and sum1]:[%.4f] [sum2]:[%.4f]  [sum3]:[%.4f] [cost:%.4f]', sum0 + lamda(1) * sum1, lamda(2) * sum2, lamda(3) * sum3, err);
    %str = sprintf('[all cost:%.6f] [sum0:%.6f] [sum1:%.6f] [sum2:%.6f] [sum3:%.6f]', err, sum0, sum1, sum2, sum3);
    %fprintf('%s', str);

    
end



