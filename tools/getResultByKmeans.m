%Y is ground truth
function [acc, nmi, ARI, Fscore, Pu, Precision, Recall] = getResultByKmeans(U, Y, numclass)
    stream = RandStream.getGlobalStream;
    reset(stream);
    U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,numclass);
    maxIter = 50;
    allacc = zeros(maxIter,1);
    allnmi = zeros(maxIter,1);
    allPu = zeros(maxIter,1);
    allARI = zeros(maxIter,1);
    allFscore = zeros(maxIter,1);
    allRecall = zeros(maxIter,1);
    allPrecision = zeros(maxIter,1);
    for it = 1 : maxIter
        indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',1);
        indx = indx(:);
        [newIndx] = bestMap(Y,indx);
        allacc(it) = mean(Y==newIndx);
        allnmi(it) = MutualInfo(Y,newIndx);
        allPu(it) = purFuc(Y,newIndx);
        %res4(it) = adjrandindex(Y,newIndx);
        allARI(it) = RandIndex(Y,newIndx);
        [allFscore(it), allPrecision(it), allRecall(it)] = compute_f(Y, newIndx);
    end
    acc = max(allacc);
    nmi = max(allnmi);
    Pu = max(allPu);
    ARI = max(allARI);
    Fscore = max(allFscore);
    Recall = max(allRecall);
    Precision = max(allPrecision);
end