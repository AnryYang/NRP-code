function [] = nrp(graph, full, directed, d, k1, k2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOADING ADJACENCY MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
is_directed = false;
if directed>0
    is_directed = true;
end

is_full = false;
if full>0
    is_full = true;
end

parent_folder = '../../';

fprintf('loading adjacency matrix, random walk matrix and out-degree list\n');

if is_full==false
    matfile = strcat(parent_folder, 'data/', graph, '/train.mat');
else
    matfile = strcat(parent_folder, 'data/', graph, '/full.mat');
end

load(matfile);
n = size(A,1);
fprintf('number of nodes: %d\n', n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MATRIX FACTORIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = 0.15;
ppa = 1-alpha;

halfd = d/2;

fprintf('SVDing\n');

start = clock;
% [U, T, V] = svds(A, halfd);
[U, T, V] = bksvd(A, halfd);
A = [];
clear A;

T = sqrt(T);
X = (U * T);
Y1 = (V * T);

elapsedTime1 = etime(clock, start);
fprintf('elapsed time for SVD is %g seconds\n', elapsedTime1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PPR UPDATING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start = clock;
[row, col] = find(Dout>0);
for j=1:numel(col)
    i = col(j);
    X(i,:) = X(i,:)./Dout(i);
end

Y = Y1;
X1 = X;
for i=2:k1
    X = ppa.* P * X + X1;
end

X = (ppa*alpha).*X;

for j=1:numel(col)
    i = col(j);
    X(i,:) = Dout(i).*X(i,:);
end

elapsedTime2 = etime(clock, start);
fprintf('elapsed time for computing ppr is %g seconds\n', elapsedTime2);

P = [];
clear P;

weighted=true;

start=clock;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REWEIGHTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if weighted==true
    fw = ones(1,n);  % forward weight
    bw = ones(1,n);  % backward weight

    h = halfd/2;
    min_w = 1/n;  % min allowed weight
    lambda = 10;  % regulization parameter

    Xsq = X.^2;  % element-wise square
    Ysq = Y.^2;  % element-wise square

    XY = zeros(1,n);
    XYsq = zeros(1,n);
    for i=1:n
        XY(i) = X(i,:)*Y(i,:)';
        XYsq(i) = Xsq(i,:)*Ysq(i,:)';
    end


    if is_directed==true
        for j=1:k2
            fprintf('%d-th iter for bwd weight\n', j);
            bw = update_dw(min_w, lambda, h, Din, Dout, bw, fw, Y, X, Ysq, Xsq, XY, XYsq);  % update backward weight
            fprintf('%d-th iter for fwd weight\n', j);
            fw = update_dw(min_w, lambda, h, Dout, Din, fw, bw, X, Y, Xsq, Ysq, XY, XYsq);  % update forward weight
        end
    else
        for j=1:k2
            fprintf('%d-th iter for bwd weight\n', j);
            bw = update_uw(min_w, lambda, h, Dout, bw, fw, Y, X, Ysq, Xsq, XY, XYsq);  % update backward weight
            fprintf('%d-th iter for fwd weight\n', j);
            fw = update_uw(min_w, lambda, h, Dout, fw, bw, X, Y, Xsq, Ysq, XY, XYsq);  % update forward weight
        end
    end

    X = bsxfun(@times,X,fw');  % multiple forward embedding vectors with their corresonding forward weights
    Y = bsxfun(@times,Y,bw');  % multiple backward embedding vectors with their corresonding backward weights
end

elapsedTime3 = etime(clock, start);
fprintf('elapsed time for all is %g seconds\n', elapsedTime1+elapsedTime2+elapsedTime3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MATERIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[a,b] = size(X);
fprintf('content embedding matrix shape: %d, %d\n', a, b);

format="bin";

if format=="bin"
    if is_full==false
        suffix = '.train.bin.src';
    else
        suffix = '.bin.src';
    end
    [a,b] = size(X);
    fprintf('content embedding matrix shape: %d, %d\n', a, b);
    xarr = reshape(X', 1, []);
    fprintf('array list length: %d, %d\n', size(xarr));

    algo='/nrp.';

    epath=strcat(parent_folder, 'embds/', graph, algo, int2str(d), suffix)
    fprintf('writing content array to %s\n', epath);
    fileID = fopen(epath,'w');
    fwrite(fileID, xarr, 'double');
    fclose(fileID);

    [a,b] = size(Y);
    fprintf('context embedding matrix shape: %d, %d\n', a, b);
    carr = reshape(Y', 1, []);
    fprintf('array list length: %d, %d\n', size(carr));

    if is_full==false
        suffix = '.train.bin.tgt';
    else
        suffix = '.bin.tgt';
    end

    epath=strcat(parent_folder, 'embds/', graph, algo, int2str(d), suffix);
    fprintf('writing context array to %s\n', epath);
    fileID = fopen(epath,'w');
    fwrite(fileID, carr, 'double');
    fclose(fileID);
else
    if is_full==false
        suffix = '_train_U.csv';
    else
        suffix = '_U.csv';
    end

    algo='_nrp';

    epath=strcat(parent_folder, 'embds/', graph, '/', graph, algo, suffix);
    fprintf('writing content embedding matrix to %s\n', epath);
    matx = X;
    writematrix(matx,epath)

    [a,b] = size(Y);
    fprintf('context embedding matrix shape: %d, %d\n', a, b);
    %carr = reshape(Y', 1, []);
    %fprintf('array list length: %d, %d\n', size(carr));

    if is_full==false
        suffix = '_train_V.csv';
    else
        suffix = '_V.csv';
    end

    epath=strcat(parent_folder, 'embds/', graph, '/', graph, algo, suffix);
    fprintf('writing context embedding matrix to %s\n', epath);
    maty = Y;
    writematrix(maty,epath)
end

end
