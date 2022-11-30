%%I dedicate this work To my son "BERGHOUT Loukmane"close all;
%%{
clear all;
clc

rng(1);
%{
ncls = 5;
nmodal = 2;
field_eye = 15;
field_narr = 5;
num_vocabulary = 789; 

Eye = csvread('Data2/Eye3.csv');
narrs = csvread('Data2/Narr3.csv');
class0 = csvread('Data2/Cat3.csv');
%}

%ncls = 2;
%nmodal = 3;
field_eye = 15;
field_narr = 1;
num_vocabulary = 789; 

Eye = csvread('Data2/gaze3.csv');
Path = csvread('Data2/paths3.csv');
narrs = csvread('Data2/tmd3.csv');
%class0 = csvread('Data2/Cat3b.csv');

MB = zeros(size(narrs,1)/field_narr,3);
for i =1:size(MB,1)
    MB(i,1:3) = narrs(i*field_narr, 1:3);
end
Tmax = max(MB(:,3)); 

nins = size(MB,1);
nobs = sum(MB(:,3));

pos = 0;
data = zeros(nobs, field_eye);
Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data(pos+1:pos+num_block,:) = (Eye((c-1)*field_eye+1:c*field_eye, 4:3+num_block))';
    Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end

% first scale data, raw data rough range is [-8,8] but most in [-1,1]
data = data / 2;

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




%% Training Options
Options.max_itera=2;            % maximum number of learning itterations
Options.N_gs=50;                  % number of gibbs samplling steps
Options.Nneurons=20;        % number of neurons in the hidden layer
Options.eps=0.01;                 % learning rate
Options.Sz_mb=500;                 % size if mini-batch of data
%% Training process
%% Load Options
eps=Options.eps;
Nneurons =Options.Nneurons;
N_gs=Options.N_gs;
max_itera=Options.max_itera;
%% initialization
% for gauss, data range is [-2,2]
I2=data;                                                % save a copy from the training data
I22 = data2;
I23 = data3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Major difference is shape, and Gaussian h (/sigma) and v (*sigma, no sigmoid), and text v (softmax)
% most hidden units inference is the same


Nneurons2 = 20;
Nneurons3 = 20;
% hyperparam for gauss
sigma = 1.;
%W = (2*rand(size(data,2),Nneurons));                    % generate randomly input wiegths  W
%v_bias = zeros(1,size(data,2));                         % initial bias in visible layer (input layer)
%h_bias = zeros(1,Nneurons);                             % initial bias in hidden layer
W1 = (1/Nneurons * rand(size(data,2),Nneurons));                    % generate randomly input wiegths  W
W2 = (1/Nneurons3 * rand(Nneurons,Nneurons3));                    % generate randomly input wiegths  W
W12 = (1/Nneurons * rand(size(data2,2),Nneurons2));                    % generate randomly input wiegths  W
W22 = (1/Nneurons3 * rand(Nneurons2,Nneurons3));                    % generate randomly input wiegths  W

W13 = (1/Nneurons * rand(size(data3,2),Nneurons2));                    % generate randomly input wiegths  W
W23 = (1/Nneurons3 * rand(Nneurons2,Nneurons3));                    % generate randomly input wiegths  W

hidden_p1 =  sigmoid(data(1:Options.Sz_mb,:) * W1 / sigma);
hidden_p12 =  sigmoid(data2(1:Options.Sz_mb,:) * W12 / sigma);
hidden_p13 =  sigmoid(data3(1:Options.Sz_mb,:) * W13 / sigma);
hidden_p2 =  sigmoid(hidden_p1 * W2 + hidden_p12 * W22 + hidden_p13 * W23);
%hidden_p2 =  sigmoid(hidden_p1 * W2 + hidden_p12 * W22);
% the following are just for initialization for vi
hidden_p20 =  0.5*ones(size(data,1), Nneurons3);
hidden_p20mb =  0.5*ones(Options.Sz_mb, Nneurons3);
%hidden_p2 =  0*sigmoid(hidden_p1 * W2);
% data shape = N*dimdata
% W1 shape = dimdata*dimh1, W2 shape = dimh1*dimh2



variationaliter = 2;
evaliter = 4;
%}

%%{
%% training processe
errvecT = zeros(1,max_itera);
for i = 1:max_itera
    if mod(i,2)==0
        disp(i);
    end
    errvec = zeros(1,N_gs);
    sampletype = 1;
    if sampletype==0 % row-of-image-level selection
        ordering = randperm(size(data,1),Options.Sz_mb);% randomly shose a batch of data
        mini_batch = data(ordering, :);                 % load our mini-Batch
        mini_batch2 = data2(ordering, :);                 % load our mini-Batch
    elseif sampletype==1 % image-level selection
        ordering = randperm(floor(size(data,1) / Options.Sz_mb),1);
        mini_batch = data((ordering-1)*Options.Sz_mb+1 : ordering*Options.Sz_mb , :);
        mini_batch2 = data2((ordering-1)*Options.Sz_mb+1 : ordering*Options.Sz_mb , :);
        mini_batch3 = data3((ordering-1)*Options.Sz_mb+1 : ordering*Options.Sz_mb , :);
    end
    
    hidden_p1 =  sigmoid(mini_batch * W1 / sigma + hidden_p20mb * (W2'));         % Find hidden units by sampling the visible layer.
    hidden_p1b = hidden_p1;
    hidden_p12 =  sigmoid(mini_batch2 * W12 / sigma + hidden_p20mb * (W22'));         % Find hidden units by sampling the visible layer.
    hidden_p12b = hidden_p12;
    hidden_p13 =  sigmoid(mini_batch3 * W13 / sigma + hidden_p20mb * (W23'));         % Find hidden units by sampling the visible layer.
    hidden_p13b = hidden_p13;
    for k = 1:variationaliter
        hidden_p2 =  sigmoid(hidden_p1b * W2 + hidden_p12b * W22 + hidden_p13b * W23);         % Find hidden units by sampling the visible layer.
        hidden_p1b =  sigmoid(mini_batch * W1 / sigma + hidden_p2 * (W2'));         % Find hidden units by sampling the visible layer.
        hidden_p12b =  sigmoid(mini_batch2 * W12 / sigma + hidden_p2 * (W22'));         % Find hidden units by sampling the visible layer.
        hidden_p13b =  sigmoid(mini_batch3 * W13 / sigma + hidden_p2 * (W23'));         % Find hidden units by sampling the visible layer.
    end
    
    for j = 1:N_gs % start gibbs sampling using energy function
        
        hidden_p2 =  sigmoid(hidden_p1b * W2 + hidden_p12b * W22);         % Find hidden units by sampling the visible layer.
        hidden_p1b =  sigmoid(mini_batch * W1 / sigma + hidden_p2 * (W2'));         % Find hidden units by sampling the visible layer.
        hidden_p12b =  sigmoid(mini_batch2 * W12 / sigma + hidden_p2 * (W22'));         % Find hidden units by sampling the visible layer.
        hidden_p13b =  sigmoid(mini_batch3 * W13 / sigma + hidden_p2 * (W23'));         % Find hidden units by sampling the visible layer.
    
        % warning: no sigmoid
        visible_p = hidden_p1b* W1' * sigma;   % Find visible units by sampling from the hidden ones.
        bp1 =  sigmoid(visible_p * W1 / sigma + hidden_p2 * (W2'));         % Find hidden units by sampling the visible layer.
        visible2_p = hidden_p12b* W12' * sigma;   % Find visible units by sampling from the hidden ones.
        bp12 =  sigmoid(visible2_p * W12 / sigma + hidden_p2 * (W22'));         % Find hidden units by sampling the visible layer.
        visible3_p = hidden_p13b* W13' * sigma;   % Find visible units by sampling from the hidden ones.
        bp13 =  sigmoid(visible3_p * W13 / sigma + hidden_p2 * (W23'));         % Find hidden units by sampling the visible layer.
        bp2 =  sigmoid(bp1 * W2 + bp12 * W22 + bp13 * W23);         % Find hidden units by sampling the visible layer.
        
        pD1 = mini_batch'*hidden_p1b / sigma ;                             % Positive Divergence
        nD1 = visible_p'*bp1 / sigma ;                           % Negative Divergence
        W1 = W1 + eps*(pD1 - nD1) / (Options.Sz_mb);                        % update weights using contrastive divergence
        pD2 = hidden_p1b'*hidden_p2;                             % Positive Divergence
        nD2 = bp1'*bp2;                           % Negative Divergence
        W2 = W2 + eps*(pD2 - nD2) / (Options.Sz_mb);                        % update weights using contrastive divergence
        
        pD12 = mini_batch2'*hidden_p12b / sigma ;                             % Positive Divergence
        nD12 = visible2_p'*bp12 / sigma ;                           % Negative Divergence
        W12 = W12 + eps*(pD12 - nD12) / (Options.Sz_mb);                        % update weights using contrastive divergence
        pD22 = hidden_p12b'*hidden_p2;                             % Positive Divergence
        nD22 = bp12'*bp2;                           % Negative Divergence
        W22 = W22 + eps*(pD22 - nD22) / (Options.Sz_mb);                        % update weights using contrastive divergence
        
        pD13 = mini_batch3'*hidden_p13b / sigma;                             % Positive Divergence
        nD13 = visible3_p'*bp13 / sigma;                           % Negative Divergence
        W13 = W13 + eps*(pD13 - nD13) / (Options.Sz_mb);                        % update weights using contrastive divergence
        pD23 = hidden_p13b'*hidden_p2;                             % Positive Divergence
        nD23 = bp13'*bp2;                           % Negative Divergence
        W23 = W23 + eps*(pD23 - nD23) / (Options.Sz_mb);                        % update weights using contrastive divergence

        
        errvec(j) =  mse((mini_batch-visible_p)) + mse((mini_batch2-visible2_p)) + mse((mini_batch-visible3_p));        % Estimate negative ll
        
    end
    errvecT(i) = mean(errvec);%training error history
end

%% training accuracy
Tr_h=sigmoid(I2* W1 / sigma + hidden_p20 * (W2'));        % calculated the visible layer
Tr2_h=sigmoid(I22* W12 / sigma + hidden_p20 * (W22'));        % calculated the visible layer
Tr3_h=sigmoid(I23* W13 / sigma + hidden_p20 * (W23'));        % calculated the visible layer
for k = 1:evaliter
    Tr_h2=sigmoid(Tr_h* W2 + Tr2_h* W22);        % calculated the visible layer
    Tr_h=sigmoid(I2* W1 / sigma + Tr_h2*(W2'));        % calculated the visible layer
    Tr2_h=sigmoid(I22* W12 / sigma + Tr_h2*(W22'));        % calculated the visible layer
    Tr3_h=sigmoid(I23* W13 / sigma + Tr_h2*(W23'));        % calculated the visible layer
end
% warning: no sigmoid
Tr_v=Tr_h* W1' * sigma ;     % calculated the hidden layer
Tr2_v=Tr2_h* W12' * sigma;     % calculated the hidden layer
Tr3_v=Tr3_h* W13' * sigma;     % calculated the hidden layer
Tr_acc = mse(I2-Tr_v)/sigma + mse(I22-Tr2_v)/sigma + mse(I23-Tr3_v)/sigma;   % Estimate negative loglikelihood
%% save trained net
net.input=I2;       % save the original normalized training data
net.regen=Tr_v;     % save the regenerated  input (reconstructed)
net.W1=W1;            % save updated weights weights
net.W2=W2;            % save updated weights weights
net.W12=W12;            % save updated weights weights
net.W13=W13;            % save updated weights weights
net.W22=W22;            % save updated weights weights
net.W23=W23;            % save updated weights weights
net.Tr_acc=Tr_acc;  % save training accracy
%net.hist=smooth(errvecT,13);   % save the smooth version of history of training error
net.hist = errvecT;

%% Illustration
plot(1:length(net.hist),net.hist,'LineWidth',2)
xlabel('number of iterations')
ylabel('RMSE of training')

% save hidden2 to file
predinput = zeros(nins,Nneurons3);
pos = 0;
for c=1:nins
    num_block = MB(c,3);
    temp = sum(Tr_h2(pos+1:pos+num_block, :));
    predinput(c,:) = temp/num_block;
    pos = pos+num_block;
end
%writematrix(predinput,'predinput3.csv');
predinput3 = predinput;

Eye = csvread('Data2/gaze2.csv');
Path = csvread('Data2/paths2.csv');
narrs = csvread('Data2/tmd2.csv');
class0 = csvread('Data2/Cat2b.csv');
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

%
%% load training parameters
W1=net.W1; % weights
W2=net.W2; % weights
W12=net.W12; % weights
W22=net.W22; % weights
W13=net.W13; % weights
W23=net.W23; % weights
%% prediction
Ts_h=sigmoid(x* W1 / sigma);        % calculated the visible layer
Ts2_h=sigmoid(x2* W12 / sigma);        % calculated the visible layer
Ts3_h=sigmoid(x3* W13 / sigma);        % calculated the visible layer
for k = 1:evaliter
    Ts_h2=sigmoid(Ts_h* W2 + Ts2_h* W22 + Ts3_h* W23);        % calculated the visible layer
    Ts_h=sigmoid(x* W1 / sigma + Ts_h2*(W2'));        % calculated the visible layer
    Ts2_h=sigmoid(x2* W12 / sigma + Ts_h2*(W22'));        % calculated the visible layer
    Ts3_h=sigmoid(x3* W13 / sigma + Ts_h2*(W23'));        % calculated the visible layer
end
% warning: no sigmoid
Ts_v=Ts_h* W1' * sigma;     % calculated the hidden layer
Ts2_v=Ts2_h* W12' * sigma;     % calculated the hidden layer
Ts3_v=Ts3_h* W13' * sigma;     % calculated the hidden layer

% save hidden2 to file
predinput = zeros(nins,Nneurons3);
pos = 0;
for c=1:nins
    num_block = MB(c,3);
    temp = sum(Ts_h2(pos+1:pos+num_block, :));
    predinput(c,:) = temp/num_block;
    pos = pos+num_block;
end
predinput2 = predinput;
%writematrix(predinput,'predinput2.csv');

%{
%save('mldbm.mat', 'x3','x2');
save('mldbm.mat');
load('mldbm.mat');

%x3 = csvread('predinput3.csv');
y3 = csvread('Data2/Cat3b.csv');
%x2 = csvread('predinput2.csv');
y2 = csvread('Data2/Cat2b.csv');

B = mnrfit(predinput3,y3);
yhat = mnrval(B,predinput2);
[~, q] = max(yhat, [], 2) ;
acc = sum(q==y2) / size(y2,1);
%}