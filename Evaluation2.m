
mode2 = 2;
lambda0 = 1;

svmprecision = 50000;
C = 2;

alphabet = ["a","b","c","d","e","f","g","h"];

clipvalue = -16;

nmodal = 3;
field_trag = 15;
field_tmd = 1;

nf1 = 10;
nf2 = 22;
nf3 = 3;
ncls = 3;
nftotal = nf1+nf2+nf3;

paths = csvread('Data2/paths2.csv');
Eye = csvread('Data2/gaze2.csv');
tmd = csvread('Data2/tmd2.csv');
class0 = csvread('Data2/Cat2b.csv');

nf0 = 5;
piprior = 10;
maxiter = 100;
nins = size(tmd,1)/field_tmd;

% store meta data of data instances
MB = zeros(size(tmd,1)/field_tmd,3);
for i =1:size(MB,1)
    MB(i,1:3) = tmd(i*field_tmd, 1:3);
end
Tmax = max(MB(:,3));

nobs = sum(MB(:,3));

% store all observations from all data instances into data1 - data3 matrix
pos = 0;
data1 = zeros(nobs, field_trag);
data2 = zeros(nobs, field_trag);
data3 = zeros(nobs, field_tmd);
Idx = zeros(nins, Tmax);
for c=1:nins
    num_block = MB(c,3);
    data1(pos+1:pos+num_block,:) = (paths((c-1)*field_trag+1:c*field_trag, 4:3+num_block))';
    data2(pos+1:pos+num_block,:) = (Eye((c-1)*field_trag+1:c*field_trag, 4:3+num_block))';
    data3(pos+1:pos+num_block,:) = (tmd((c-1)*field_tmd+1:c*field_tmd, 4:3+num_block))';
    Idx(c,1:num_block) = pos+1:pos+num_block;
    pos = pos+num_block;
end



load('script/model2.mat');
load('script/mynested2.mat');

% initialize pattern assignments
assign1 = rand(nobs,nf1);
for i=1:nobs
    assign1(i,:) = assign1(i,:) / sum(assign1(i,:));
end
assign2 = rand(nobs,nf2);
for i=1:nobs
    assign2(i,:) = assign2(i,:) / sum(assign2(i,:));
end
assign3 = rand(nobs,nf3);
for i=1:nobs
    assign3(i,:) = assign3(i,:) / sum(assign3(i,:));
end


assign0 = rand(nobs,nf0);
for i=1:nobs
    assign0(i,:) = assign0(i,:) / sum(assign0(i,:));
end





bool0 = ones(ncls,nins);
for i=1:nins
    bool0(class0(i,1),i) = 0;
end
boolvec0 = reshape(bool0,[1,ncls*nins]);

A0 = zeros(nins,nins*ncls);
for i=1:nins
    A0(i,(i-1)*ncls+1:i*ncls) = boolvec0(1,(i-1)*ncls+1:i*ncls);
end

assign = horzcat(assign1,assign2,assign3);

burnin = 1;
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

if mode2==1
    aq = aq0;
    bq = bq0*ones(1,nftotal);
elseif mode2==2
    sq = lambda0*ones(1,nftotal);
    zetasquare = ones(ncls,nftotal);
end


for iter=1:10
    % assign patterns
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
        for pat=1:nf2
            temp(1,pat) = - sum(0.5*log(varq2(pat,:))) -0.5*sum(((data2(i,:)-muq2(pat,:)).^2)./varq2(pat,:));
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier2(i,pat);
            end
        end
        temp = temp - max(temp);
        temp(temp<clipvalue) = clipvalue;
        temp = exp(temp);
        assign2(i,:) = temp/sum(temp);
    end
    for i=1:nobs
        temp = zeros(1,nf3);
        state = assign0(i,:);
        for pat=1:nf3
            temp(1,pat) = - sum(0.5*log(varq3(pat,:))) -0.5*sum(((data3(i,:)-muq3(pat,:)).^2)./varq3(pat,:));
            temp(1,pat) = temp(1,pat) - log(state*theta_prob{1,3}(:,pat));
            if burnin==0
                temp(1,pat) = temp(1,pat) * multiplier3(i,pat);
            end
        end
        temp = temp - max(temp);
        temp(temp<clipvalue) = clipvalue;
        temp = exp(temp);
        assign3(i,:) = temp/sum(temp);
    end
    
    % assign states
    pos = 0;
    for c=1:nins
        num_block = MB(c,3);
        assigns = cell(1,nmodal);
        assigns{1,1} = assign1(pos+1:pos+num_block,:);
        assigns{1,2} = assign2(pos+1:pos+num_block,:);
        assigns{1,3} = assign3(pos+1:pos+num_block,:);
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
            usernn = 1;
            if usernn==1
                rnninput = (probseq(1:t-1,:))';
                rnnoutput = predict(mynested2,rnninput);
                temp = rnnoutput(:,end);
                prob = prob.*(temp');
            end
            probseq(t,:) = prob / sum(prob);
        end
        
        pos = pos + num_block;
    end
    
end

% prepare data instance representation for class prediction
assign = horzcat(assign1,assign2,assign3);

etavisual = reshape(eta, nftotal,ncls)';

predinput = zeros(nins,nftotal);
pos = 0;
for c=1:nins
    num_block = MB(c,3);
    temp = sum(assign(pos+1:pos+num_block, :));
    predinput(c,:) = temp/num_block;
    pos = pos+num_block;
end


% predict class
scores = etavisual*predinput';
predcls = zeros(1,nins);
for c=1:nins
    [~,cls] = max(scores(:,c));
    predcls(1,c) = cls;
end
fprintf('Accuracy = %f\n', sum(predcls==class0')/nins);
fprintf('Predicted class (instance 1-15): \n');
disp(predcls(1:15));
fprintf('True class (instance 1-15): \n');
disp((class0(1:15))');
