%%I dedicate this work To my son "BERGHOUT Loukmane"close all;
%%{

clear all;
clc

rng(1);

ncls = 2;
nmodal = 3;
field_eye = 15;
field_narr = 1;
num_vocabulary = 789;

readdata=1;
if readdata==1
    Eye = csvread('Data2/gaze3.csv');
    Path = csvread('Data2/paths3.csv');
    narrs = csvread('Data2/tmd3.csv');
    classraw = csvread('Data2/Cat3b.csv');
else
    Eye1 = csvread('Data1/Eye3.csv');
    Eye2 = csvread('Data1/Eye2.csv');
    Eye = [Eye1;Eye2];
    narrs1 = csvread('Data1/Narr3.csv');
    narrs2 = csvread('Data1/Narr2.csv');
    narrs = [narrs1;narrs2];
    class01 = csvread('Data1/Cat3.csv');
    class02 = csvread('Data1/Cat2.csv');
    classraw = [class01;class02];
end




MB = zeros(size(narrs,1)/field_narr,3);
for i =1:size(MB,1)
    MB(i,1:3) = narrs(i*field_narr, 1:3);
end
Tmax = max(MB(:,3));

nins = size(MB,1);
nobs = sum(MB(:,3));

pos = 0;
data = zeros(nobs, field_eye);
%Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data(pos+1:pos+num_block,:) = (Eye((c-1)*field_eye+1:c*field_eye, 4:3+num_block))';
    %Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end
% first scale data, raw data rough range is [-8,8] but most in [-1,1]
data = data / 2;


pos = 0;
data2 = zeros(nobs, field_eye);
%Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data2(pos+1:pos+num_block,:) = (Path((c-1)*field_eye+1:c*field_eye, 4:3+num_block))';
    %Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end
% first scale data, raw data rough range is [-8,8] but most in [-1,1]
data2 = data2 / 2;


pos = 0;
data3 = zeros(nobs, field_narr);
%Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data3(pos+1:pos+num_block,:) = (narrs((c-1)*field_narr+1:c*field_narr, 4:3+num_block))';
    %Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end



%{
data2numerical = zeros(nobs,1);
pos = 0;
for c=1:nins
    num_block = MB(c,3);
    data2numerical(pos+1:pos+num_block,:) = (narrs((c-1)*field_narr+3, 4:3+num_block))';
    pos = pos+num_block;
end
data2 = false(nobs, num_vocabulary);
for i=1:nobs
    data2(i, data2numerical(i)) = 1;
end
%}



class0 = zeros(nobs,1);
pos = 0;
for c=1:nins
    num_block = MB(c,3);
    class0(pos+1:pos+num_block) = classraw(c);
    pos = pos+num_block;
end


%% Training Options
Options.max_itera=5;            % maximum number of learning itterations
Options.N_gs=50;                  % number of gibbs samplling steps
Options.Nneurons=25;        % number of neurons in the hidden layer
Options.eps=0.01;                 % learning rate
Options.Sz_mb=2000;                 % size if mini-batch of data
%% Training process
%% Load Options
eps=Options.eps;
Nneurons =Options.Nneurons;
N_gs=Options.N_gs;
max_itera=Options.max_itera;
%% initialization

%% initialization
% for gauss, data range is [-2,2]
I2=data;                                                % save a copy from the training data
I22 = data2;
I23 = data3;

nftotal = Options.Nneurons;
C = 2;
C2 = 1/2;


% initialize V
V = 1/Options.Nneurons * ones(ncls, Options.Nneurons);

warning('off','all');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Major difference is shape, and Gaussian h (/sigma) and v (*sigma, no sigmoid), and text v (softmax)
% most hidden units inference is the same


% hyperparam for gauss
sigma = 0.5;
scale = 1e-8;
%W = (2*rand(size(data,2),Nneurons));                    % generate randomly input wiegths  W
%v_bias = zeros(1,size(data,2));                         % initial bias in visible layer (input layer)
%h_bias = zeros(1,Nneurons);                             % initial bias in hidden layer
W1 = (1/Nneurons * rand(size(data,2),Nneurons));                    % generate randomly input wiegths  W
W12 = (1/Nneurons * rand(size(data2,2),Nneurons));                    % generate randomly input wiegths  W
W13 = (1/Nneurons * rand(size(data3,2),Nneurons));                    % generate randomly input wiegths  W
% data shape = N*dimdata
% W1 shape = dimdata*dimh1, W2 shape = dimh1*dimh2



variationaliter = 5;
evaliter = 10;
%}


%% training processe
errvecT = zeros(1,max_itera);
for i = 1:max_itera
    if mod(i,5)==0
        disp(i);
    end
    errvec = zeros(1,N_gs);
    sampletype = 0;
    if sampletype==0 % row-of-image-level selection
        ordering = randperm(size(data,1),Options.Sz_mb);% randomly shose a batch of data
        mini_batch = data(ordering, :);                 % load our mini-Batch
        mini_batch2 = data2(ordering, :); 
        mini_batch3 = data3(ordering, :); 
        labelsnow = class0(ordering); % load our mini-Batch
    elseif sampletype==1 % image-level selection
        ordering = randperm(floor(size(data,1) / Options.Sz_mb),1);
        mini_batch = data((ordering-1)*Options.Sz_mb+1 : ordering*Options.Sz_mb , :);
        mini_batch2 = data2((ordering-1)*Options.Sz_mb+1 : ordering*Options.Sz_mb , :);
        mini_batch3 = data3((ordering-1)*Options.Sz_mb+1 : ordering*Options.Sz_mb , :);
        labelsnow = class0((ordering-1)*Options.Sz_mb+1 : ordering*Options.Sz_mb , :);
    end
    
    for j = 1:N_gs % start gibbs sampling using energy function
        hidden_p =  sigmoid(mini_batch * W1 / sigma + mini_batch2 * W12 /sigma + mini_batch3 * W13 /sigma);         % Find hidden units by sampling the visible layer.
        % warning: no sigmoid
        visible_p = hidden_p* W1' * sigma;   % Find visible units by sampling from the hidden ones.
        visible2_p = hidden_p* W12' * sigma;   % Find visible units by sampling from the hidden ones.
        visible3_p = hidden_p* W13' * sigma;   % Find visible units by sampling from the hidden ones.
        
        bp =  sigmoid(visible_p * W1 + visible2_p * W12 + visible3_p * W13);         % Find hidden units by sampling the visible layer.
        
        pD1 = mini_batch'*hidden_p / sigma ;                             % Positive Divergence
        nD1 = visible_p'*bp / sigma ;                           % Negative Divergence
        W1 = W1 + eps*(pD1 - nD1) / (Options.Sz_mb);                        % update weights using contrastive divergence
        pD12 = mini_batch2'*hidden_p / sigma ;                             % Positive Divergence
        nD12 = visible2_p'*bp / sigma ;                           % Negative Divergence
        W12 = W12 + eps*(pD12 - nD12) / (Options.Sz_mb);                        % update weights using contrastive divergence
        pD13 = mini_batch3'*hidden_p / sigma ;                             % Positive Divergence
        nD13 = visible3_p'*bp / sigma ;                           % Negative Divergence
        W13 = W13 + eps*(pD13 - nD13) / (Options.Sz_mb);                        % update weights using contrastive divergence
        
        % additional influence from svm
        dw1 = zeros(size(W1));
        dw12 = zeros(size(W12));
        dw13 = zeros(size(W13));
        for ins = 1:size(mini_batch,1)
            x = mini_batch(ins,:);
            x2 = mini_batch2(ins,:);
            x3 = mini_batch3(ins,:);
            h = hidden_p(ins,:);
            % retrieve ground truth label
            labeltrue = labelsnow(ins);
            % find 2nd prediction's corresponding class
            scores = h*(V');
            scoretrue = scores(labeltrue);
            scores(labeltrue) = 1000;
            [out,idx] = sort(scores);
            label2nd = idx(end-1);
            score2nd = out(end-1);
            % if scoreture-score2nd>1, it is not support vector, just ignore it in calculation
            % if <1, we encourage score difference to be large and close to 1
            if scoretrue-score2nd < 1
                % V shape dimh*dimcls, hidden shape 1*dimh, W shape diminput*dimh
                % derivative is (x') * (Vtrue-Vcandidate).* (logistic.*(1-logistic))
                temp = (V(labeltrue,:)-V(label2nd,:)).*h.*(1-h);
                dw1 = dw1 + scale*x' * temp;
                dw12 = dw12 + scale*x2' * temp;
                dw13 = dw13 + scale*x3' * temp;
            end
        end
        % original contrastive divergence is gradient ascend to maximize loglikelihood
        % to encourage score difference to be large and close to 1, it is additive not substractive
        % thie is different from zhujun's paper, because they use (0or1 - score1st)
        % empirically verified, you should use +
        W1 = W1 + eps*C2*dw1;
        W12 = W12 + eps*C2*dw12;
        W13 = W13 + eps*C2*dw13;
        
        errvec(j) =  mse((mini_batch-visible_p)) + mse((mini_batch3-visible3_p)) + mse((mini_batch2-visible2_p));        % Estimate negative ll
        
    end
    errvecT(i) = mean(errvec);%training error history
end

%% prepare variables for SVM
% prepare auxiliary variables for maximum-margin learning
bool0 = ones(ncls,nins);
for i=1:nins
    bool0(classraw(i,1),i) = 0;
end
boolvec0 = reshape(bool0,[1,ncls*nins]);

A0 = zeros(nins,nins*ncls);
for i=1:nins
    A0(i,(i-1)*ncls+1:i*ncls) = boolvec0(1,(i-1)*ncls+1:i*ncls);
end

Tr_h=sigmoid(I2* W1 / sigma + I22* W12 / sigma + I23* W13 / sigma);        % calculated the visible layer
occurs = zeros(nins, nftotal);
pos = 0;
for ins = 1:nins
    num_block = MB(ins,3);
    temp = Tr_h(pos+1:pos+num_block,:);
    occurs(ins,:) = sum(temp)/num_block;
    pos = pos+num_block;
end

fdy = zeros(ncls*nftotal,nins);
X0 = zeros(nftotal,nins);
for ins = 1:nins
    occur = occurs(ins,:);
    idx = classraw(ins,1);
    
    for c=1:ncls
        fdy( (c-1)*nftotal+1:c*nftotal, (ins-1)*ncls+c) = -occur';
        fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) = fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) + occur';
    end
    X0(:,ins) = occur';
end

% constraints left side, all constraints are A=[A0;-A0;-diag(boolvec0)]
% matlab quadprog is min_x 0.5*x'Hx + f'x, s.t. Ax<=b
% here x is v^{(l)}(y)
% score function coeff eta = sum(instance, y != true y) v^{(l)}(y) * delta f^{(l)}(y)
% constraint 1: sum(y != true y)per instance v^{(l)}(y)<C
% constraint 2: sum(y != true y)per instance v^{(l)}(y)>0
% constraint 3: any v^{(l)}(y)>0

% from zhujun paper, f(y,h) is vector where (y-1)K+1:yK is h and others =0
% V is just reshaping score function coeff eta to vector
% R_hinge loss is at the end of page 4, it is different from medlda
% for y = ytruth, the two terms of R_hinge are both 0
% for y != ytruth, 1st term = 1
% 2nd term is actually selecting another y whose score prediction is close to 1
% but in your paper you encourage ytruth's score - other y's score >=1 generally
% so you can change the R_hinge loss

% the key difference from zhujun paper is that you use mean h
% first, you still need to get the argmax y for your own R_hinge
% which is essentially finding the columns in fdy
% then just take partial derivative wrt R_hinge to update W1 and W12

%H = diag(boolvec0)*(fdy')*diag(repmat(bq,1,ncls) / aq)*fdy*diag(boolvec0);
%scaler = aq/sum(bq);
H = diag(boolvec0)*(fdy')*fdy*diag(boolvec0);
% just to supress warning message, otherwise it always indicates 'not symmetric'
H = (H+H')/2;
scaler = 2/Options.Sz_mb;
A=[A0;-A0;-diag(boolvec0)];
b=[C*ones(nins,1);zeros(nins,1);zeros(nins*ncls,1)];
Aeq=[];
beq=[];
lb=[];
ub=[];
muvecinitial = scaler*ones(1,ncls*nins);
%muvec = quadprog(H,-boolvec0,A,b,Aeq,beq,lb,ub);
% surpress warning message
svmoptions = optimoptions(@quadprog, 'Display','off');
x0 = []; % ignore initial point
muvec = quadprog(H,-boolvec0,A,b,Aeq,beq,lb,ub,x0,svmoptions);

mu = reshape(muvec', ncls,nins);
eta = zeros(1,ncls*nftotal);

for ins=1:nins
    tc = classraw(ins,1);
    for c=1:ncls
        if c~=tc
            %eta = eta + (mu(c,i) * (fdy(:,ncls*(i-1)+c))').* repmat(bq,1,ncls) / aq ;
            eta = eta + (mu(c,ins) * (fdy(:,ncls*(ins-1)+c))');
        end
    end
end
V = reshape(eta, nftotal,ncls)';



%% training accuracy
Tr_h=sigmoid(I2* W1 / sigma + I22* W12 / sigma + I23* W13 / sigma);        % calculated the visible layer
% warning: no sigmoid
Tr_v=Tr_h* W1' * sigma ;     % calculated the hidden layer
Tr2_v=Tr_h* W12' * sigma ;     % calculated the hidden layer
Tr3_v=Tr_h* W13' * sigma ;     % calculated the hidden layer
Tr_acc = mse(I2-Tr_v)/sigma + mse(I22-Tr2_v)/sigma + mse(I23-Tr3_v)/sigma;   % Estimate negative loglikelihood
%% save trained net
net.input=I2;       % save the original normalized training data
net.regen=Tr_v;     % save the regenerated  input (reconstructed)
net.W1=W1;            % save updated weights weights
net.W12=W12;            % save updated weights weights
net.W13=W13;            % save updated weights weights
net.Tr_acc=Tr_acc;  % save training accracy
%net.hist=smooth(errvecT,13);   % save the smooth version of history of training error
net.hist = errvecT;

%{
% check training class prediction
scores = occurs * (V');
[~,predcls] = max(scores,[],2);

match = sum(predcls==classraw);
acc = match/nins;
fprintf('accuracy = %f\n', acc);
%}

%save('llsl.mat', 'V','net');
save('llsl.mat');

%%{
load llsl;


%% Prediction


Eye = csvread('Data2/gaze2.csv');
Path = csvread('Data2/paths2.csv');
narrs = csvread('Data2/tmd2.csv');
classraw = csvread('Data2/Cat2b.csv');
%{
Eye = csvread('Data1/Eye2.csv');
narrs = csvread('Data1/Narr2.csv');
class0 = csvread('Data1/Cat2.csv');
%}
nins = size(narrs,1)/field_narr;
MB = zeros(size(narrs,1)/field_narr,3);
for i =1:size(MB,1)
    MB(i,1:3) = narrs(i*field_narr, 1:3);
end
Tmax = max(MB(:,3));

nobs = sum(MB(:,3));

pos = 0;
data1b = zeros(nobs, field_eye);
Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data1b(pos+1:pos+num_block,:) = (Eye((c-1)*field_eye+1:c*field_eye, 4:3+num_block))';
    Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end
%{
data2numerical = zeros(nobs,1);
pos = 0;
for c=1:nins
    num_block = MB(c,3);
    data2numerical(pos+1:pos+num_block,:) = (narrs((c-1)*field_narr+3, 4:3+num_block))';
    pos = pos+num_block;
end
data2b = false(nobs, num_vocabulary);
for i=1:nobs
    data2b(i, data2numerical(i)) = 1;
end
%}
pos = 0;
data2b = zeros(nobs, field_eye);
%Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data2b(pos+1:pos+num_block,:) = (Path((c-1)*field_eye+1:c*field_eye, 4:3+num_block))';
    %Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end
pos = 0;
data3b = zeros(nobs, field_narr);
%Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data3b(pos+1:pos+num_block,:) = (narrs((c-1)*field_narr+1:c*field_narr, 4:3+num_block))';
    %Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end

x = data1b / 2;
x2 = data2b / 2;
x3 = data3b;


%% load training parameters
W1=net.W1; % weights
W12=net.W12; % weights
W13=net.W13; % weights
%% prediction
Ts_h=sigmoid(x* W1 / sigma + x2* W12 /sigma + x3* W13 / sigma);        % calculated the visible layer
% warning: no sigmoid
Ts_v=Ts_h* W1' * sigma;     % calculated the hidden layer
Ts2_v=Ts_h* W12' * sigma;     % calculated the hidden layer
Ts3_v=Ts_h* W13' * sigma;     % calculated the hidden layer

occurs = zeros(nins, nftotal);
pos = 0;
for ins = 1:nins
    num_block = MB(ins,3);
    temp = Ts_h(pos+1:pos+num_block,:);
    occurs(ins,:) = sum(temp)/num_block;
    pos = pos+num_block;
end
% test accuracy is not accurate, because text part is incorrectly generated
scores = occurs * (V');
[~,predcls] = max(scores,[],2);

match = sum(predcls==classraw);
acc = match/nins;
fprintf('accuracy = %f\n', acc);

%}