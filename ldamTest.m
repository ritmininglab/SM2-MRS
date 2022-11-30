%%{
clear all;
rng(0);

narrs1 = csvread('Data1/Narr3.csv');
narrs2 = csvread('Data1/Narr2.csv');
narrs = [narrs1;narrs2];
field_narr = 5;
Tmax_narr = max(narrs(:,3)); %%%%%%% this is max number of words in a sequence
num_chain = size(narrs,1)/field_narr;
num_chain1 = size(narrs1,1)/field_narr;
num_chain2 = size(narrs2,1)/field_narr;


maxIter = 1000;

lag_narr = 0;
alpha2 = 1/2; % since now doc-phase-topic count is limited, alpha2 should be smaller than phase-topic case
igamma2 = 1/8;
%consider_stickbreak=0;

num_trunk = 5;
num_vocabulary = 789; % 651 for data1, 510-764 for data2

num_topic = 40;
a0 = 1/40; % Dirichlet prior for topic-word relationship, smaller - more sparse, tested 0.1 best
b0 = 2; % Dirichlet prior for doc-topic relationship, smaller - more sparse
c0 = 10; % transition ones matrix
d0 = 0; % self transition diagonal matrix

% initialization topic assignment
topic_assign = zeros(num_chain, Tmax_narr);
for c=1:num_chain
    num_block = narrs(c*field_narr,3);
    %topic_assign(c,4+lag_narr:num_block+3) = floor(1+(num_topic-1)*rand(1, num_block-lag_narr));
    %topic_assign(c,4+lag_narr:num_block+3) = floor(1+(num_topic-0.1)*(1:num_block-lag_narr)/(num_block-lag_narr));
    topic_assign(c,1:num_block) = floor(1+(num_topic-0.1)*(1:num_block)/num_block);
    %topic_assign(c,4+lag_narr-1) = 1; % used to calculate u2
end
topic_prob_vec = cell(num_chain,Tmax_narr);

warning('off','all');

%%%%%%%% topic parameter initialization
tau = a0*ones(num_topic, num_vocabulary);
tau_prob = zeros(num_topic, num_vocabulary);
% count occurrence of words in each topic
for doc = 1:num_chain
    %image = topic_assign(doc,1);
    %people = narrs(doc*field_narr,2);
    num_block = narrs(doc*field_narr,3);
    for pos = 4:3+num_block
        word = narrs(doc*field_narr-2,pos);
        topic = topic_assign(doc,pos-3);
        tau(topic,word) = tau(topic,word)+1;
    end
end
for idx = 1:num_topic
    tau_prob(idx,:) = tau(idx,:) / sum(tau(idx,:));
    %tau_prob(idx,:) = dirichlet_sample(tau(idx,:));
end


% count occurrence of topic in all docs
phi = zeros(num_chain, num_topic);
phi_prob = zeros(num_chain, num_topic);
for doc = 1:num_chain
    temp = topic_assign(doc,:);
    for topic = 1:num_topic
        phi(doc, topic) = b0 + sum(sum(temp == topic));
    end
    phi_prob(doc,:) = phi(doc,:) / sum(phi(doc,:));
    % do we need dirichlet sample?
end

%}






%Eye = csvread('PCAMatrix.csv');
%Idx2PosMatrix = csvread('Idx2PosMatrix.csv');
Eye1 = csvread('Data1/Eye3.csv');
Eye2 = csvread('Data1/Eye2.csv');
Eye = [Eye1;Eye2];

autoregress = 0;

field_eye = 15;
field_pca = 15;
startfield = 1;
field_ar = field_pca; % number of total_fields for auto-regression input vector
lag_eye = 0;

% currently initialize with prior, will be refined very soon
% by doing so, we can keep the following update of mixmu and patmu the same as those in main algorithm
% and make pat params the same as global prior and never update them, then there is no hierarchy
% now make sk = s0, remove S00 and invS00

% note that they are scalar var, not std
sx = 1; % used for SM1 and SM2 approximation only
sk = 100; % used for sampling tail mix theta only

df0 = field_ar + 1000; % affect inverse wishart mean
% notice that max pcavar could be as large as 5, but most pcavar < 3
% if not consider mix, best setting is S0=1*, kappa0 = 0.5*
% if consider mix, best setting is S0=0.5*, kappa0 = 0.5*, S00 = 4*eye
S0 = sx *(df0-field_ar-1)*eye(field_ar);  % prior for response noise, and also prior for A noise among row
% scale up first pc's prior var
%S0(1,1) = S0(1,1)*1;
kappa0 = sx/sk; % prior sigma = sigmax / kappa0

invS0 = inv(S0 / (df0-field_ar-1) ); % it is not the simple inverse of S0, as S0 is IW param

% store meta data
Tmax = max(Eye(:,3));
MB = zeros(num_chain,3);
for i =1:num_chain
    MB(i,1:3) = Eye(i*field_pca, 1:3);
end
nmap = sum(MB(:,3));

mu0pca = zeros(field_pca,1);


pos = 1;
response = zeros(nmap, field_pca);
Idx = zeros(num_chain, Tmax);
for c=1:num_chain
    num_block = MB(c,3);
    %%%%%%%%%%%%%%%%%%%%%% choose total_fields
    response(pos:pos+num_block-lag_eye-1,:) = (Eye((c-1)*field_pca+1:c*field_pca, 4+lag_eye:3+num_block))';
    %response(pos:pos+num_block-lag_eye-1,:) = (Eye((c-1)*field_eye+2:(c-1)*field_eye+5, 4+lag_eye:3+num_block))';
    Idx(c,1:num_block) = pos:pos+num_block-lag_eye-1;
    pos = pos+num_block-lag_eye;
end

% empty you matrix, so code will not change for
you = response;

upperlimit = 32;

% store number of mixture per pattern
mix_meta = zeros(upperlimit,1);
% changed to 2 mix per pattern
mix_meta(1:num_topic,1) = 1*ones(num_topic,1);

% store number of mixture per pattern
mix_meta = zeros(upperlimit,1);
% changed to 2 mix per pattern
mix_meta(1:num_topic,1) = 1*ones(num_topic,1);

% initialize mixture assign
% no need because here we only consider pat, so all mix assign are 1 and never updated
mix_assign = zeros(num_chain, Tmax);
for c=1:num_chain
    num_block = MB(c,3);
    % 1:lag_eye column are empty
    mix_assign(c,1+lag_eye:num_block) = ones(1, num_block-lag_eye);
    % this randomly initialize mix = 1 or 2
    %mix_assign(c,1+lag_eye:num_block) = floor(1+(rand(1,num_block-lag_eye)>0.5));
end

% store count of observation per mixture
mix_count = zeros(upperlimit,upperlimit);
% store dp mixture stick portions for each pattern
mix_beta = cell(upperlimit,1);

match1 = zeros(num_chain*160,1);
pos = 1;
% concatenate each row of pattern assign to the match vector
for c=1:num_chain
    num_block = MB(c,3);
    match1(pos:pos+num_block-lag_eye-1,:) = (topic_assign(c, lag_eye+1:num_block))';
    pos = pos+num_block-lag_eye;
end
match1 = match1(1:pos-1,:);
match2 = zeros(num_chain*160,1);
pos = 1;
% concatenate each row of pattern assign to the match vector
for c=1:num_chain
    num_block = MB(c,3);
    match2(pos:pos+num_block-lag_eye-1,:) = (mix_assign(c, lag_eye+1:num_block))';
    pos = pos+num_block-lag_eye;
end
match2 = match2(1:pos-1,:);

% store pattern parameters in mix_param etc
% the major difference from NIW hierarchy is that now mix_param connects observation and prior, no hierarchy

% store mixture mean
mix_param = cell(upperlimit,1); % mu
mix_S = cell(upperlimit,1); % intermediate variable to calculate sigma
mix_sigmainv = cell(upperlimit,1); % inverse sigma
mix_count = cell(upperlimit,1);

% currently initialize with prior, will be refined very soon
% by doing so, we can keep the following update of mixmu and patmu the same as those in main algorithm
% and make pat params the same as global prior and never update them, then there is no hierarchy
% now make sk = s0, remove S00 and invS00, replace mu0pca with mu0pca

for p=1:num_topic
    for mix=1:1
        match3 = (match1==p); % .*(match2==mix);
        response_p = response(match3==1,:);
        response_p = response_p';
        mix_count{p,mix} = size(response_p,2); % number of events assigned to this pattern and mixture
        if mix_count{p,mix}>0
            n = mix_count{p,mix};
            xbar = mean(response_p,2);
            temp = response_p - repmat(xbar,1,mix_count{p,mix});
            scattermat = temp * temp';
            mix_param{p,mix} = kappa0/(kappa0+mix_count{p,mix}) * mu0pca ...
                + mix_count{p,mix}/(kappa0+mix_count{p,mix}) * xbar; % horizontal 1-by-2 vector
            mix_S{p,mix} = S0 + scattermat + kappa0*n / (kappa0+n) * (xbar-mu0pca)*(xbar-mu0pca)';
            mix_sigmainv{p,mix} = inv(mix_S{p,mix} / (n+df0-field_ar-1));
            % posterior var of mixmu is sigma/kappan, so if there are 10 obs, posterior var is really small already
        else
            mix_param{p,mix} = mu0pca + sk*randn(field_pca, 1);
            mix_S{p,mix} = S0;
            mix_sigmainv{p,mix} = invS0;
        end
    end
end





for iter=1:0
    
    if mod(iter,100)==0
        fprintf('Iter %d \n', iter);
    end
    
    % assign topic
    for doc = 1:num_chain
        image = MB(doc,1);
        people = MB(doc,2);
        num_block = MB(doc,3);
        
        for pos = 1:num_block
            word = narrs(doc*field_narr-2,pos+3);
            r1 = (phi(doc,:))' .* tau_prob(:, word);
            
            % Idx do not contain meta 3 columns, and do not contain lag_eye
            idx_you = Idx(doc,pos);
            response_now = response(idx_you,:)';
            
            r2 = zeros(num_topic, 1);
            for pattern = 1:num_topic
                mix = 1;
                r2(pattern,1) = 1000 * sqrt(det(mix_sigmainv{p,mix}))...
                    *exp(-0.5*(response_now - mix_param{p,mix})' * mix_sigmainv{p,mix} * (response_now - mix_param{p,mix}));
            end
            r = r1 .* r2;
            
            r = r ./ sum(r);
            topic_assign(doc,pos) = 1+sum(rand() > cumsum(r));
            topic_prob_vec{doc,pos} = r;
        end
    end
    
    % update tau and phi
    tau = a0*ones(num_topic, num_vocabulary);
    tau_prob = zeros(num_topic, num_vocabulary);
    % count occurrence of words in each topic
    for doc = 1:num_chain
        %image = MB(doc,1);
        %people = MB(doc,2);
        num_block = MB(doc,3);
        for pos = 4:3+num_block
            word = narrs(doc*field_narr-2,pos);
            topic = topic_assign(doc,pos-3);
            tau(topic,word) = tau(topic,word)+1;
        end
    end
    for idx = 1:num_topic
        tau_prob(idx,:) = tau(idx,:) / sum(tau(idx,:));
        %tau_prob(idx,:) = dirichlet_sample(tau(idx,:));
    end
    
    phi = zeros(num_chain, num_topic);
    phi_prob = zeros(num_chain, num_topic);
    for doc = 1:num_chain
        temp = topic_assign(doc,:);
        for topic = 1:num_topic
            phi(doc, topic) = b0 + sum(sum(temp == topic));
        end
        phi_prob(doc,:) = phi(doc,:) / sum(phi(doc,:));
        % do we need dirichlet sample?
    end
    
    
    % update mix params
    
    match1 = zeros(num_chain*160,1);
    pos = 1;
    % concatenate each row of pattern assign to the match vector
    for c=1:num_chain
        num_block = MB(c,3);
        match1(pos:pos+num_block-lag_eye-1,:) = (topic_assign(c, lag_eye+1:num_block))';
        pos = pos+num_block-lag_eye;
    end
    match1 = match1(1:pos-1,:);
    match2 = zeros(num_chain*160,1);
    pos = 1;
    % concatenate each row of pattern assign to the match vector
    for c=1:num_chain
        num_block = MB(c,3);
        match2(pos:pos+num_block-lag_eye-1,:) = (mix_assign(c, lag_eye+1:num_block))';
        pos = pos+num_block-lag_eye;
    end
    match2 = match2(1:pos-1,:);
    
    % store pattern parameters in mix_param etc
    % the major difference from NIW hierarchy is that now mix_param connects observation and prior, no hierarchy
    
    % store mixture mean
    mix_param = cell(upperlimit,1); % mu
    mix_S = cell(upperlimit,1); % intermediate variable to calculate sigma
    mix_sigmainv = cell(upperlimit,1); % inverse sigma
    
    % currently initialize with prior, will be refined very soon
    % by doing so, we can keep the following update of mixmu and patmu the same as those in main algorithm
    % and make pat params the same as global prior and never update them, then there is no hierarchy
    % now make sk = s0, remove S00 and invS00, replace mu0pca with mu0pca
    
    for p=1:num_topic
        for mix=1:1
            match3 = (match1==p); % .*(match2==mix);
            response_p = response(match3==1,:);
            response_p = response_p';
            mix_count{p,mix} = size(response_p,2); % number of events assigned to this pattern and mixture
            if mix_count{p,mix}>0
                n = mix_count{p,mix};
                xbar = mean(response_p,2);
                temp = response_p - repmat(xbar,1,mix_count{p,mix});
                scattermat = temp * temp';
                mix_param{p,mix} = kappa0/(kappa0+mix_count{p,mix}) * mu0pca ...
                    + mix_count{p,mix}/(kappa0+mix_count{p,mix}) * xbar; % horizontal 1-by-2 vector
                mix_S{p,mix} = S0 + scattermat + kappa0*n / (kappa0+n) * (xbar-mu0pca)*(xbar-mu0pca)';
                mix_sigmainv{p,mix} = inv(mix_S{p,mix} / (n+df0-field_ar-1));
                % posterior var of mixmu is sigma/kappan, so if there are 10 obs, posterior var is really small already
            else
                mix_param{p,mix} = mu0pca + sk*randn(field_pca, 1);
                mix_S{p,mix} = S0;
                mix_sigmainv{p,mix} = invS0;
            end
        end
    end
    
    
end


counts3 = zeros(num_topic,1);
for i =1:num_topic
    counts3(i,1) = sum(sum(topic_assign==i));
end
wordcounts = sum(tau)';

topic_topwords = zeros(num_topic,num_vocabulary);
for i=1:num_topic
    [value,idx] = sort(tau(i,:),'descend');
    topic_topwords(i,:) = idx;
end


occur = zeros(num_chain, num_topic);
topicfreq = zeros(num_chain, num_topic);

% store meta data;
MB = zeros(size(narrs,1)/field_narr,3);
for i =1:size(MB,1)
    MB(i,1:3) = narrs((i-1)*field_narr+1, 1:3);
end

for doc = 1:num_chain
    num_block = MB(doc,3);
    for t = 1:num_block %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%12/20 found error, shall not be 1+1:num_block+3
        topic = topic_assign(doc,t);
        occur(doc, topic) = occur(doc, topic)+1;
    end
    topicfreq(doc, :) = occur(doc, :) / MB(doc,3);
end

% prepare input matrix
%csvwrite('inputfull-LDAMC1.csv', [MB, topicfreq]);


%save('ldam_allvariables.mat');
%save('ldam.mat', 'tau','tau_prob','phi','phi_prob','B','mix_param','mix_S','mix_sigmainv','B');

load('ldam.mat');


for iter=1:maxIter
    
    if mod(iter,100)==0
        fprintf('Iter %d \n', iter);
    end
    
    % assign topic
    for doc = 1:num_chain
        image = MB(doc,1);
        people = MB(doc,2);
        num_block = MB(doc,3);
        
        for pos = 1:num_block
            word = narrs(doc*field_narr-2,pos+3);
            r1 = (phi(doc,:))' .* tau_prob(:, word);
            
            % Idx do not contain meta 3 columns, and do not contain lag_eye
            idx_you = Idx(doc,pos);
            response_now = response(idx_you,:)';
            
            r2 = zeros(num_topic, 1);
            for pattern = 1:num_topic
                mix = 1;
                r2(pattern,1) = 1000 * sqrt(det(mix_sigmainv{p,mix}))...
                    *exp(-0.5*(response_now - mix_param{p,mix})' * mix_sigmainv{p,mix} * (response_now - mix_param{p,mix}));
            end
            r = r1 .* r2;
            
            r = r ./ sum(r);
            topic_assign(doc,pos) = 1+sum(rand() > cumsum(r));
            topic_prob_vec{doc,pos} = r;
        end
    end
    
    
end



occur = zeros(num_chain, num_topic);
topicfreq = zeros(num_chain, num_topic);

% store meta data;
MB = zeros(size(narrs,1)/field_narr,3);
for i =1:size(MB,1)
    MB(i,1:3) = narrs((i-1)*field_narr+1, 1:3);
end

for doc = 1:num_chain
    num_block = MB(doc,3);
    for t = 1:num_block %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%12/20 found error, shall not be 1+1:num_block+3
        topic = topic_assign(doc,t);
        occur(doc, topic) = occur(doc, topic)+1;
    end
    topicfreq(doc, :) = occur(doc, :) / MB(doc,3);
end



input3 = topicfreq(1:num_chain1,:);

y3 = csvread('Data1/Cat3.csv');
y2 = csvread('Data1/Cat2.csv');
y = [y3;y2];

%B = mnrfit(input3,y3);

%B = mnrfit(topicfreq,y);

input2 = topicfreq(num_chain1+1:end,:);
yhat = mnrval(B,input2);
[~, q] = max(yhat, [], 2) ;
acc = sum(q==y2) / size(y2,1);
fprintf('Accuracy = %f\n', acc);