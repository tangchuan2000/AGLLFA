 close all; clear all;
 addpath(genpath('.'));

DBDIR = './Dataset/';

i= 1;
DataName{i} = '3sources'; i = i + 1;
%DataName{i} = 'WebKB'; i = i + 1;
dbNum = length(DataName);

maxIter = 50;
bnorm = 1;
knn = 15; %number of neighbours for construct SIG
testLamda = 0; 
lamda = [10000  10  10];    

%%%%%%%%%%%%%%%%%%%%%%result
acc = 0;
nmi = 0;
ARI = 0;
Fscore = 0;
Pu = 0;
Precision = 0;
Recall = 0;

%%%%%%%%%%%%%%%%%%%%%%
for dbIndex = 1:dbNum
   clear X gt ;
   dbnamePre = DataName{dbIndex}; 
   dbfilename = sprintf('%s%s.mat',DBDIR,dbnamePre);
   load(dbfilename);   
   la1 = [100 500 1000 10000];   
   la2 = [10 0.1 0.01 0.001];
   la3 = [10 0.1 0.01 0.001];
   if (testLamda == 1)
        maxIter = 20;
        for m = 1:length(la1)
            for i = 1:length(la2)
                for j = 1:length(la3)
                    lamda(1) = la1(m);
                    lamda(2) = la2(i);
                    lamda(3) = la3(j);                       
                    [acc, nmi, ARI, Fscore, Pu, Precision, Recall] = AGLLFSR(X, bnorm, knn, gt, maxIter, lamda);
                    printResult(acc, nmi, ARI, Fscore, Pu, Precision, Recall, dbnamePre, lamda)
                end
            end
        end      

    else
        [acc, nmi, ARI, Fscore, Pu, Precision, Recall] = AGLLFSR(X, bnorm, knn, gt, maxIter, lamda);             
        printResult(acc, nmi, ARI, Fscore, Pu, Precision, Recall, dbnamePre, lamda)
   end    
    
end
fprintf('\n complete... \n');

function printResult(acc, nmi, ARI, Fscore, Pu, Precision, Recall, dbnamePre, lamda)
    str = sprintf('[%s ][ACC %.2f] [NMI %.2f] [ARI %.2f] [F-score %.2f] [Purity %.2f] [Precision %.2f] [Recall %.2f][lamda1:%.3f lamda2:%.3f lamda3:%.3f]  %s ',...
            dbnamePre, acc * 100, nmi * 100, ARI * 100, Fscore * 100, Pu * 100, Precision * 100, Recall * 100, lamda(1), lamda(2), lamda(3), GetTimeStrForLog());
    fprintf('%s\n',str);    
end


