rng(1);
mode2 = 2;
lambda0 = 1;
maxiter = 10;
svmprecision = 50000;
C = 2;
piprior = 10;
alphabet = ["a","b","c","d","e","f","g","h"];

clipvalue = -16;

ncls = 3;
%nmodal = 2;
field_eye = 15;
%field_narr = 5;
num_vocabulary = 789;
nf1 = 16;
nftotal = nf1;

warning('off','all');

readdata=1;
if readdata==1
    Eye = csvread('Data2/Gaze3.csv');
    %narrs = csvread('Data1/Narr3.csv');
    class0 = csvread('Data2/Cat3b.csv');
else
    Eye1 = csvread('Data1/Eye3.csv');
    Eye2 = csvread('Data1/Eye2.csv');
    Eye = [Eye1;Eye2];
    class01 = csvread('Data1/Cat3.csv');
    class02 = csvread('Data1/Cat2.csv');
    class0 = [class01;class02];
end

nf0 = 5;
nins = size(Eye,1)/field_eye;

a0 = 1; % drichlet prior for states
a2 = 1/10; % dirichlet prior for topics
mu0 = 0; % Gaussian prior mean for gaze
var0 = 1; % Gaussian prior variance for gaze

% store meta data of data instances
MB = zeros(size(Eye,1)/field_eye,3);
for i =1:size(MB,1)
    MB(i,1:3) = Eye(i*field_eye, 1:3);
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

Idx2 = zeros(nobs,1);
pos = 0;
for c=1:nins
    num_block = MB(c,3);
    Idx2(pos+1:pos+num_block) = c;
    pos = pos+num_block;
end


% initialize pattern assignments
assign1 = rand(nobs,nf1);
for i=1:nobs
    assign1(i,:) = assign1(i,:) / sum(assign1(i,:));
end


% initialize pattern parameters
muq1 = zeros(nf1,field_eye);
varq1= zeros(nf1,field_eye);
varq2= zeros(nf1,field_eye);

for pat=1:nf1
    nk = sum(assign1(:,pat));
    temp = bsxfun(@times, data1, assign1(:,pat));
    tempmu = sum(temp,1) / nk;
    
    %temp2 = bsxfun(@minus, data1, tempmu);
    %temp2 = temp2.^2;
    %temp2 = bsxfun(@times, temp2, assign1(:,pat));
    %tempvar = sum(temp2,1) / nk;
    
    varq1(pat,:) = var0;
    tempweight = 1/(nk+1);
    muq1(pat,:) = tempweight.*mu0 + (1-tempweight).*tempmu;
    
    temp2 = bsxfun(@minus, data1, muq1(pat,:));
    temp2 = temp2.^2;
    temp2 = bsxfun(@times, temp2, assign1(:,pat));
    varq2(pat,:) = sqrt(sum(temp2,1) / nk);
    
end



maxitergrad = 10;
maxitermain = 50;
burnin = 1;


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

assign = assign1;


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




% start burn-in training

for iter=1:maxiter
    if mod(iter,20)==0
        disp(iter);
    end
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
            
            
            pos = pos + num_block;
        end
    end
    
    % assign patterns and topics
    for i=1:nobs
        temp = zeros(1,nf1);
        %state = assign0(i,:);
        for pat=1:nf1
            
            
            temp(1,pat) = - sum(0.5*log(varq1(pat,:))) -0.5*sum(((data1(i,:)-muq1(pat,:)).^2)./varq1(pat,:));
            %temp(1,pat) = temp(1,pat) - log(state*theta_prob{1,1}(:,pat));
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier1(i,pat);
            end
        end
        temp = temp - max(temp);
        temp(temp<clipvalue) = clipvalue;
        temp = exp(temp);
        assign1(i,:) = temp/sum(temp);
    end
    
    
    % update pattern parameters and topic parameters
    for pat=1:nf1
        nk = sum(assign1(:,pat));
        if nk>0
            temp = bsxfun(@times, data1,assign1(:,pat));
            tempmu = sum(temp,1) / nk;
            
            %temp2 = bsxfun(@minus, data1, tempmu);
            %temp2 = temp2.^2;
            %temp2 = bsxfun(@times, temp2, assign1(:,pat));
            %tempvar = sum(temp2,1) / nk;
            
            varq1(pat,:) = var0;
            tempweight = 1/(nk+1);
            muq1(pat,:) = tempweight.*mu0 + (1-tempweight).*tempmu;
            temp2 = bsxfun(@minus, data1, muq1(pat,:));
            temp2 = temp2.^2;
            temp2 = bsxfun(@times, temp2, assign1(:,pat));
            varq2(pat,:) = sqrt(sum(temp2,1) / nk);
            
        end
    end
    
    
    
    
    
end


% prepare auxiliary variables for maximum-margin learning

assign = assign1;


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

% formal training

burin = 0;

for iter=1:maxitermain
    if mod(iter,20)==0
       disp(iter);
    end
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
            
            
            pos = pos + num_block;
        end
    end
    
    % assign patterns and topics
    for i=1:nobs
        temp = zeros(1,nf1);
        %state = assign0(i,:);
        for pat=1:nf1
            
            
            temp(1,pat) = - sum(0.5*log(varq1(pat,:))) -0.5*sum(((data1(i,:)-muq1(pat,:)).^2)./varq1(pat,:));
            %temp(1,pat) = temp(1,pat) - log(state*theta_prob{1,1}(:,pat));
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier1(i,pat);
            end
        end
        temp = temp - max(temp);
        temp(temp<clipvalue) = clipvalue;
        temp = exp(temp);
        assign1(i,:) = temp/sum(temp);
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
    
    
    
    % prepare auxiliary variables for maximum-margin learning
    
    assign = assign1;
    
    
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










mu = reshape(muvec', ncls,nins);
eta = zeros(1,ncls*nftotal);

for ins=1:nins
    tc = class0(ins,1);
    for c=1:ncls
        if c~=tc
            %eta = eta + (mu(c,i) * (fdy(:,ncls*(i-1)+c))').* repmat(bq,1,ncls) / aq ;
            eta = eta + (mu(c,ins) * (fdy(:,ncls*(ins-1)+c))');
        end
    end
end
etavisual = reshape(eta, nftotal,ncls)';

save('med_gaze.mat', 'varq1','muq1','varq2','etavisual');

%{
load('med_gaze.mat');

for iter=1:10
    for i=1:nobs
        temp = zeros(1,nf1);
        for pat=1:nf1
            
            temp(1,pat) = - sum(0.5*log(varq1(pat,:))) -0.5*sum(((data1(i,:)-muq1(pat,:)).^2)./varq1(pat,:));
        end
        temp = temp - max(temp);
        temp(temp<clipvalue) = clipvalue;
        temp = exp(temp);
        assign1(i,:) = temp/sum(temp);
    end
    
end


assign = assign1;

predinput = zeros(nins,nftotal);

pos = 0;
for c=1:nins
    num_block = MB(c,3);
    temp = sum(assign(pos+1:pos+num_block, :));
    predinput(c,:) = temp/num_block;
    pos = pos+num_block;
end

scores = etavisual*predinput';
predcls = zeros(1,nins);
for c=1:nins
    [~,cls] = max(scores(:,c));
    predcls(1,c) = cls;
end
fprintf('Accuracy = %f\n', sum(predcls==class0')/nins);

%}
