rng(1);
mode2 = 2;
lambda0 = 1;
maxiter = 1000;
svmprecision = 50000;
C = 2;
piprior = 10;
alphabet = ["a","b","c","d","e","f","g","h"];

clipvalue = -16;

ncls = 5;
nmodal = 2;
field_eye = 15;
field_narr = 5;
num_vocabulary = 789;
nf1 = 16;
nf2 = 20;
nftotal = nf1+nf2;

Eye = csvread('Data1/Eye3.csv');
narrs = csvread('Data1/Narr3.csv');
class0 = csvread('Data1/Cat3.csv');

nf0 = 5;
nins = size(narrs,1)/field_narr;

a0 = 1; % drichlet prior for states
a2 = 1/10; % dirichlet prior for topics
mu0 = 0; % Gaussian prior mean for gaze
var0 = 1; % Gaussian prior variance for gaze

% store meta data of data instances
MB = zeros(size(narrs,1)/field_narr,3);
for i =1:size(MB,1)
    MB(i,1:3) = narrs(i*field_narr, 1:3);
end
Tmax = max(MB(:,3));

nobs = sum(MB(:,3));

% store all observations from all data instances into data1 and data2 matrix
pos = 0;
data1 = zeros(nobs, field_eye);
Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data1(pos+1:pos+num_block,:) = (Eye((c-1)*field_eye+1:c*field_eye, 4:3+num_block))';
    Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end

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



% initialize pattern assignments
assign1 = rand(nobs,nf1);
for i=1:nobs
    assign1(i,:) = assign1(i,:) / sum(assign1(i,:));
end
assign2 = rand(nobs,nf2);
for i=1:nobs
    assign2(i,:) = assign2(i,:) / sum(assign2(i,:));
end


% initialize pattern parameters
muq1 = zeros(nf1,field_eye);
varq1= zeros(nf1,field_eye);

for pat=1:nf1
    nk = sum(assign1(:,pat));
    temp = bsxfun(@times, data1, assign1(:,pat));
    tempmu = sum(temp,1) / nk;
    
    temp2 = bsxfun(@minus, data1, tempmu);
    temp2 = temp2.^2;
    temp2 = bsxfun(@times, temp2, assign1(:,pat));
    tempvar = sum(temp2,1) / nk;
    
    varq1(pat,:) = var0;
    tempweight = 1/(nk+1);
    muq1(pat,:) = tempweight.*mu0 + (1-tempweight).*tempmu;
    
    temp2 = bsxfun(@minus, data1, muq1(pat,:));
    temp2 = temp2.^2;
    temp2 = bsxfun(@times, temp2, assign1(:,pat));
    varq1(pat,:) = sqrt(sum(temp2,1) / nk);
    
end

% initialize topic parameters
tau = zeros(nf2, num_vocabulary);
tau_prob = zeros(nf2, num_vocabulary);
for pat=1:nf2
    temp = bsxfun(@times, data2,assign2(:,pat));
    tau(pat,:) = a2 + sum(temp,1);
    tau_prob(pat,:) = tau(pat,:) / sum(tau(pat,:));
end


% initialize state assignments
assign0 = rand(nobs,nf0);
for i=1:nobs
    assign0(i,:) = assign0(i,:) / sum(assign0(i,:));
end
probs = cell(1,nmodal);

ncluster = [nf1,nf2];
dataraw = cell(1,nmodal);
data1hot = cell(1,nmodal);

dataraw{1,1} = assign1;
dataraw{1,2} = assign2;

theta = cell(1,nmodal);
theta_prob = cell(1,nmodal);
for i = 1:nmodal
    theta{1,i} = zeros(nf0, ncluster(i));
    theta_prob{1,i} = zeros(nf0, ncluster(i));
end

for i = 1:nmodal
    for pat = 1:nf0
        temp = bsxfun(@times, dataraw{1,i}, assign0(:,pat));
        theta{1,i}(pat,:) = a0 + sum(temp,1);
        theta_prob{1,i}(pat,:) = theta{1,i}(pat,:) / sum(theta{1,i}(pat,:));
    end
end


maxitergrad = 10;
maxiterter = 1000;
burnin = 1;

% set up network architecture
nhidden = 32;
layers = [ ...
    sequenceInputLayer(nf0)
    gruLayer(nhidden,'OutputMode','sequence')
    fullyConnectedLayer(nf0)
    softmaxLayer
    classificationLayer];
gpuDevice()
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',50, ...
    'GradientThreshold',1, ...
    'Verbose',0, ...
    'MiniBatchSize',128);

% prepare auxiliary variables for maximum-margin learning
bool0 = ones(ncls,nins);
for i=1:nins
    bool0(class0(i,1),i) = 0;
end
boolvec0 = reshape(bool0,[1,ncls*nins]);

A0 = zeros(nins,nins*ncls);
for i=1:nins
    A0(i,(i-1)*ncls+1:i*ncls) = boolvec0(1,(i-1)*ncls+1:i*ncls);
end

assign = horzcat(assign1,assign2);


occurs = zeros(nins, nftotal);
pos = 0;
for ins = 1:nins
    num_block = MB(ins,3);
    temp = assign(pos+1:pos+num_block,:);
    occurs(ins,:) = sum(temp)/num_block;
    pos = pos+num_block;
end

fdy = zeros(ncls*nftotal,nins);
X0 = zeros(nftotal,nins);
for ins = 1:nins
    occur = occurs(ins,:);
    idx = class0(ins,1);
    
    for c=1:ncls
        fdy( (c-1)*nftotal+1:c*nftotal, (ins-1)*ncls+c) = -occur';
        fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) = fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) + occur';
    end
    X0(:,ins) = occur';
end

% prepare group-wise regularization parameters
if mode2==1
    aq = aq0;
    bq = bq0*ones(1,nftotal);
elseif mode2==2
    sq = lambda0*ones(1,nftotal);
    zetasquare = ones(ncls,nftotal);
end




% start training

for iter=1:maxiterter
    
    if burnin==0
        multiplier1 = zeros(nobs,nf1);
        multiplier2 = zeros(nobs,nf2);
        pos = 0;
        for j=1:nins
            num_block = MB(j,3);
            for pat=1:nf1
                addon1 = 0;
                for cla=1:ncls
                    if cla==class0(j,1)
                        continue;
                    end
                    addon1 = addon1 + mu(cla,j)*(eta(1,(class0(j,1)-1)*nftotal+pat) - eta(1,(cla-1)*nftotal+pat));
                end
                multiplier1(pos+1:pos+num_block,pat) = exp(addon1/num_block);
            end
            
            for pat=1:nf2
                addon2 = 0;
                for cla=1:ncls
                    if cla==class0(j,1)
                        continue;
                    end
                    addon2 = addon2 + mu(cla,j)*(eta(1,(class0(j,1)-1)*nftotal+pat+nf1) - eta(1,(cla-1)*nftotal+pat+nf1));
                end
                multiplier2(pos+1:pos+num_block,pat) = exp(addon2/num_block);
            end
            
            pos = pos + num_block;
        end
    end
    
    % assign patterns and topics
    for i=1:nobs
        temp = zeros(1,nf1);
        state = assign0(i,:);
        for pat=1:nf1
            
            
            temp(1,pat) = - sum(0.5*log(varq1(pat,:))) -0.5*sum(((data1(i,:)-muq1(pat,:)).^2)./varq1(pat,:));
            temp(1,pat) = temp(1,pat) - log(state*theta_prob{1,1}(:,pat));
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier1(i,pat);
            end
        end
        temp = temp - max(temp);
        temp(temp<clipvalue) = clipvalue;
        temp = exp(temp);
        assign1(i,:) = temp/sum(temp);
    end
    for i=1:nobs
        temp = zeros(1,nf2);
        state = assign0(i,:);
        word = data2numerical(i);
        temptheta = bsxfun(@minus, psi(theta{1,2}), psi(sum(theta{1,2}, 2)));
        
        for pat=1:nf2
            
            temptau = psi(tau(pat,word)) - psi(sum(tau(pat,:)));
            temp(1,pat) = exp(temptau) * (state* exp(temptheta(:,pat)));
            
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier2(i,pat);
            end
        end
        assign2(i,:) = temp/sum(temp);
    end
    
    % assign states
    pos = 0;
    for c=1:nins
        num_block = MB(c,3);
        assigns = cell(1,nmodal);
        assigns{1,1} = assign1(pos+1:pos+num_block,:);
        assigns{1,2} = assign2(pos+1:pos+num_block,:);
        wthetas = cell(1,nmodal);
        probseq = zeros(num_block, nf0);
        assign0now = zeros(num_block, nf0);
        
        t=1;
        prob = 1000*ones(1,nf0);
        for i = 1:nmodal
            wthetas{1,i} = bsxfun(@times, theta_prob{1,i}, assigns{1,i}(t,:));
            temp = sum(wthetas{1,i},2);
            prob = prob.*(temp');
        end
        probseq(t,:) = prob / sum(prob);
        
        for t=2:num_block
            prob = 1000*ones(1,nf0);
            for i = 1:nmodal
                wthetas{1,i} = bsxfun(@times, theta_prob{1,i}, assigns{1,i}(t,:));
                temp = sum(wthetas{1,i},2);
                prob = prob.*(temp');
            end
            usernn = 0;
            if usernn==1
                rnninput = (probseq(1:t-1,:))';
                rnnoutput = predict(mynested,rnninput);
                temp = rnnoutput(:,end);
                prob = prob.*(temp');
            end
            probseq(t,:) = prob / sum(prob);
        end
        
        pos = pos + num_block;
    end
    
    % update pattern parameters and topic parameters
    for pat=1:nf1
        nk = sum(assign1(:,pat));
        if nk>0
            temp = bsxfun(@times, data1,assign1(:,pat));
            tempmu = sum(temp,1) / nk;
            
            temp2 = bsxfun(@minus, data1, tempmu);
            temp2 = temp2.^2;
            temp2 = bsxfun(@times, temp2, assign1(:,pat));
            tempvar = sum(temp2,1) / nk;
            
            varq1(pat,:) = var0;
            tempweight = 1/(nk+1);
            muq1(pat,:) = tempweight.*mu0 + (1-tempweight).*tempmu;
            temp2 = bsxfun(@minus, data1, muq1(pat,:));
            temp2 = temp2.^2;
            temp2 = bsxfun(@times, temp2, assign1(:,pat));
            varq1(pat,:) = sqrt(sum(temp2,1) / nk);
            
        end
    end
    for pat=1:nf2
        temp = bsxfun(@times, data2,assign2(:,pat));
        tau(pat,:) = a2 + sum(temp,1);
        tau_prob(pat,:) = tau(pat,:) / sum(tau(pat,:));
    end
    
    % update state parameters
    for i = 1:nmodal
        for pat = 1:nf0
            temp = bsxfun(@times, dataraw{1,i}, assign0(:,pat));
            theta{1,i}(pat,:) = a0 + sum(temp,1);
            theta_prob{1,i}(pat,:) = theta{1,i}(pat,:) / sum(theta{1,i}(pat,:));
        end
    end
    
    
    pos = 0;
    xtrain = cell(nins,1);
    ytrain = cell(nins,1);
    for i = 1:nins
        num_block = MB(i,3);
        xtrain{i,1} = (assign0(pos+1:pos+num_block-1,:))';
        temp = (assign0(pos+2:pos+num_block,:))';
        [~,samplef0] = max(temp,[],1);
        ytrain{i,1} = categorical(samplef0);
        pos = pos+num_block;
    end
    
    % update network parameters
    
    mynested = trainNetwork(xtrain,ytrain,layers,options);
    save mynested;
    
    
    
    % prepare auxiliary variables for maximum-margin learning
    
    assign = horzcat(assign1,assign2);
    
    
    occurs = zeros(nins, nftotal);
    pos = 0;
    for ins = 1:nins
        num_block = MB(ins,3);
        temp = assign(pos+1:pos+num_block,:);
        occurs(ins,:) = sum(temp)/num_block;
        pos = pos+num_block;
    end
    
    fdy = zeros(ncls*nftotal,nins);
    X0 = zeros(nftotal,nins);
    for ins = 1:nins
        occur = occurs(ins,:);
        idx = class0(ins,1);
        
        for c=1:ncls
            fdy( (c-1)*nftotal+1:c*nftotal, (ins-1)*ncls+c) = -occur';
            fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) = fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) + occur';
        end
        X0(:,ins) = occur';
    end
    % maximum margin learning
    if mode2==1
        H = diag(boolvec0)*(fdy')*diag(repmat(bq,1,ncls) / aq)*fdy*diag(boolvec0);
        scaler = aq/sum(bq);
    elseif mode2==2
        H = (diag(boolvec0)*(fdy'))*diag(repmat(sq,1,ncls))*(fdy*diag(boolvec0));
        scaler = 1/sum(sq);
    end
    A=[A0;-A0;-diag(boolvec0)];
    b=[C*ones(nins,1);zeros(nins,1);zeros(nins*ncls,1)];
    Aeq=[];
    beq=[];
    lb=[];
    ub=[];
    muvecinitial = scaler*ones(1,ncls*nins);
    muvec = quadprog(H,-boolvec0,A,b,Aeq,beq,lb,ub);
    
    
    
    
    % update scoring functions and group-wise sparse posterior parameters
    mu = reshape(muvec', ncls,nins);
    eta = zeros(1,ncls*nftotal);
    
    if mode2==1
        for i=1:nins
            tc = class0(i,1);
            for c=1:ncls
                if c~=tc
                    eta = eta + (mu(c,i) * (fdy(:,ncls*(i-1)+c))').* repmat(bq,1,ncls) / aq ;
                end
            end
        end
        
        aq = a0+ncls/2;
        for f=1:nftotal
            bq(1,f) = b0;
            for c=1:ncls
                bq(1,f) = bq(1,f)+0.5 * eta(1,(c-1)*nftotal+f)^2;
            end
        end
        
    elseif mode2==2
        for i=1:nins
            tc = class0(i,1);
            for c=1:ncls
                if c~=tc
                    eta = eta + (mu(c,i) * (fdy(:,ncls*(i-1)+c))').* repmat(sq,1,ncls) ;
                end
            end
        end
        for i=1:nftotal
            g = sqrt(lambda0 * sum(zetasquare(:,i)));
            k = (ncls-1)/2;
            temp1 = 1;
            for j=1:k+1
                temp1 = temp1+factorial(k+j+1)/(factorial(k+1-j)*factorial(j)* (2*g)^j);
            end
            temp2 = 1;
            for j=1:k
                temp2 = temp2+factorial(k+j)/(factorial(k-j)*factorial(j)* (2*g)^j);
            end
            sq(1,i) = (g/lambda0)*(temp2/temp1);
        end
        for i=1:ncls
            zetasquare(i,:) = (eta(1,(i-1)*nftotal+1:i*nftotal)).^2+sq;
        end
    end
    
end


burin = 0;



for iter=1:maxiterter
    
    % evaluate influence from supervised task in pattern discovery
    if burnin==0
        multiplier1 = zeros(nobs,nf1);
        multiplier2 = zeros(nobs,nf2);
        pos = 0;
        for j=1:nins
            num_block = MB(j,3);
            for pat=1:nf1
                addon1 = 0;
                for cla=1:ncls
                    if cla==class0(j,1)
                        continue;
                    end
                    addon1 = addon1 + mu(cla,j)*(eta(1,(class0(j,1)-1)*nftotal+pat) - eta(1,(cla-1)*nftotal+pat));
                end
                multiplier1(pos+1:pos+num_block,pat) = exp(addon1/num_block);
            end
            
            for pat=1:nf2
                addon2 = 0;
                for cla=1:ncls
                    if cla==class0(j,1)
                        continue;
                    end
                    addon2 = addon2 + mu(cla,j)*(eta(1,(class0(j,1)-1)*nftotal+pat+nf1) - eta(1,(cla-1)*nftotal+pat+nf1));
                end
                multiplier2(pos+1:pos+num_block,pat) = exp(addon2/num_block);
            end
            
            pos = pos + num_block;
        end
    end
    
    % update pattern assignment
    for i=1:nobs
        temp = zeros(1,nf1);
        state = assign0(i,:);
        for pat=1:nf1
            
            
            temp(1,pat) = - sum(0.5*log(varq1(pat,:))) -0.5*sum(((data1(i,:)-muq1(pat,:)).^2)./varq1(pat,:));
            temp(1,pat) = temp(1,pat) - log(state*theta_prob{1,1}(:,pat));
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier1(i,pat);
            end
        end
        temp = temp - max(temp);
        temp(temp<clipvalue) = clipvalue;
        temp = exp(temp);
        assign1(i,:) = temp/sum(temp);
    end
    for i=1:nobs
        temp = zeros(1,nf2);
        state = assign0(i,:);
        word = data2numerical(i);
        temptheta = bsxfun(@minus, psi(theta{1,2}), psi(sum(theta{1,2}, 2)));
        
        for pat=1:nf2
            
            temptau = psi(tau(pat,word)) - psi(sum(tau(pat,:)));
            temp(1,pat) = exp(temptau) * (state* exp(temptheta(:,pat)));
            
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier2(i,pat);
            end
        end
        assign2(i,:) = temp/sum(temp);
    end
    
    % update state assignment
    pos = 0;
    for c=1:nins
        num_block = MB(c,3);
        assigns = cell(1,nmodal);
        assigns{1,1} = assign1(pos+1:pos+num_block,:);
        assigns{1,2} = assign2(pos+1:pos+num_block,:);
        wthetas = cell(1,nmodal);
        probseq = zeros(num_block, nf0);
        assign0now = zeros(num_block, nf0);
        
        t=1;
        prob = 1000*ones(1,nf0);
        for i = 1:nmodal
            wthetas{1,i} = bsxfun(@times, theta_prob{1,i}, assigns{1,i}(t,:));
            temp = sum(wthetas{1,i},2);
            prob = prob.*(temp');
        end
        probseq(t,:) = prob / sum(prob);
        
        for t=2:num_block
            prob = 1000*ones(1,nf0);
            for i = 1:nmodal
                wthetas{1,i} = bsxfun(@times, theta_prob{1,i}, assigns{1,i}(t,:));
                temp = sum(wthetas{1,i},2);
                prob = prob.*(temp');
            end
            usernn = 0;
            if usernn==1
                rnninput = (probseq(1:t-1,:))';
                rnnoutput = predict(mynested,rnninput);
                temp = rnnoutput(:,end);
                prob = prob.*(temp');
            end
            probseq(t,:) = prob / sum(prob);
        end
        
        pos = pos + num_block;
    end
    
    % update pattern and topic parameters
    for pat=1:nf1
        nk = sum(assign1(:,pat));
        if nk>0
            temp = bsxfun(@times, data1,assign1(:,pat));
            tempmu = sum(temp,1) / nk;
            
            temp2 = bsxfun(@minus, data1, tempmu);
            temp2 = temp2.^2;
            temp2 = bsxfun(@times, temp2, assign1(:,pat));
            tempvar = sum(temp2,1) / nk;
            
            varq1(pat,:) = var0;
            tempweight = 1/(nk+1);
            muq1(pat,:) = tempweight.*mu0 + (1-tempweight).*tempmu;
            temp2 = bsxfun(@minus, data1, muq1(pat,:));
            temp2 = temp2.^2;
            temp2 = bsxfun(@times, temp2, assign1(:,pat));
            varq1(pat,:) = sqrt(sum(temp2,1) / nk);
            
        end
    end
    for pat=1:nf2
        temp = bsxfun(@times, data2,assign2(:,pat));
        tau(pat,:) = a2 + sum(temp,1);
        tau_prob(pat,:) = tau(pat,:) / sum(tau(pat,:));
    end
    % update state parameters
    for i = 1:nmodal
        for pat = 1:nf0
            temp = bsxfun(@times, dataraw{1,i}, assign0(:,pat));
            theta{1,i}(pat,:) = a0 + sum(temp,1);
            theta_prob{1,i}(pat,:) = theta{1,i}(pat,:) / sum(theta{1,i}(pat,:));
        end
    end
    
    
    pos = 0;
    xtrain = cell(nins,1);
    ytrain = cell(nins,1);
    for i = 1:nins
        num_block = MB(i,3);
        xtrain{i,1} = (assign0(pos+1:pos+num_block-1,:))';
        temp = (assign0(pos+2:pos+num_block,:))';
        [~,samplef0] = max(temp,[],1);
        ytrain{i,1} = categorical(samplef0);
        pos = pos+num_block;
    end
    
    
    % update network parameters
    mynested = trainNetwork(xtrain,ytrain,layers,options);
    save mynested;
    
    
    % prepare auxiliary variables for maximum-margin learning
    
    assign = horzcat(assign1,assign2);
    
    
    occurs = zeros(nins, nftotal);
    pos = 0;
    for ins = 1:nins
        num_block = MB(ins,3);
        temp = assign(pos+1:pos+num_block,:);
        occurs(ins,:) = sum(temp)/num_block;
        pos = pos+num_block;
    end
    
    fdy = zeros(ncls*nftotal,nins);
    X0 = zeros(nftotal,nins);
    for ins = 1:nins
        occur = occurs(ins,:);
        idx = class0(ins,1);
        
        for c=1:ncls
            fdy( (c-1)*nftotal+1:c*nftotal, (ins-1)*ncls+c) = -occur';
            fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) = fdy( (idx-1)*nftotal+1:idx*nftotal, (ins-1)*ncls+c) + occur';
        end
        X0(:,ins) = occur';
    end
    % maximum-margin learning
    
    if mode2==1
        H = diag(boolvec0)*(fdy')*diag(repmat(bq,1,ncls) / aq)*fdy*diag(boolvec0);
        scaler = aq/sum(bq);
    elseif mode2==2
        H = (diag(boolvec0)*(fdy'))*diag(repmat(sq,1,ncls))*(fdy*diag(boolvec0));
        scaler = 1/sum(sq);
    end
    A=[A0;-A0;-diag(boolvec0)];
    b=[C*ones(nins,1);zeros(nins,1);zeros(nins*ncls,1)];
    Aeq=[];
    beq=[];
    lb=[];
    ub=[];
    muvecinitial = scaler*ones(1,ncls*nins);
    muvec = quadprog(H,-boolvec0,A,b,Aeq,beq,lb,ub);
    
    
    
    
    % update scoring function and group-wise regularization posterior parameters
    mu = reshape(muvec', ncls,nins);
    eta = zeros(1,ncls*nftotal);
    
    if mode2==1
        for i=1:nins
            tc = class0(i,1);
            for c=1:ncls
                if c~=tc
                    eta = eta + (mu(c,i) * (fdy(:,ncls*(i-1)+c))').* repmat(bq,1,ncls) / aq ;
                end
            end
        end
        
        aq = a0+ncls/2;
        for f=1:nftotal
            bq(1,f) = b0;
            for c=1:ncls
                bq(1,f) = bq(1,f)+0.5 * eta(1,(c-1)*nftotal+f)^2;
            end
        end
        
    elseif mode2==2
        for i=1:nins
            tc = class0(i,1);
            for c=1:ncls
                if c~=tc
                    eta = eta + (mu(c,i) * (fdy(:,ncls*(i-1)+c))').* repmat(sq,1,ncls) ;
                end
            end
        end
        for i=1:nftotal
            g = sqrt(lambda0 * sum(zetasquare(:,i)));
            k = (ncls-1)/2;
            temp1 = 1;
            for j=1:k+1
                temp1 = temp1+factorial(k+j+1)/(factorial(k+1-j)*factorial(j)* (2*g)^j);
            end
            temp2 = 1;
            for j=1:k
                temp2 = temp2+factorial(k+j)/(factorial(k-j)*factorial(j)* (2*g)^j);
            end
            sq(1,i) = (g/lambda0)*(temp2/temp1);
        end
        for i=1:ncls
            zetasquare(i,:) = (eta(1,(i-1)*nftotal+1:i*nftotal)).^2+sq;
        end
    end
    
end
