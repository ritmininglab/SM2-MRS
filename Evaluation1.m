
mode2 = 2;
lambda0 = 1;

svmprecision = 50000;
C = 2;

alphabet = ["a","b","c","d","e","f","g","h"];

clipvalue = -16;

nmodal = 2;
field_eye = 15;
field_narr = 5;
num_vocabulary = 789;

nf1 = 16;
nf2 = 20;
ncls = 5;
nftotal = nf1+nf2;

Eye = csvread('Data1/Eye2.csv');
narrs = csvread('Data1/Narr2.csv');
class0 = csvread('Data1/Cat2.csv');

nf0 = 5;
nins = size(narrs,1)/field_narr;
piprior = 10;
maxiter = 100;

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


load('script/model1.mat');
load('script/mynested.mat');


assign0 = ones(nobs,nf0);
for i=1:nobs
    assign0(i,:) = assign0(i,:) / sum(assign0(i,:));
end


for iter=1:10
    % assign patterns and topics
    for i=1:nobs
        temp = zeros(1,nf1);
        state = assign0(i,:);
        for pat=1:nf1
            
            
            temp(1,pat) = - sum(0.5*log(varq1(pat,:))) -0.5*sum(((data1(i,:)-muq1(pat,:)).^2)./varq1(pat,:));
            temp(1,pat) = temp(1,pat) - log(state*theta_prob{1,1}(:,pat));
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
        
        for pat=1:nf2
            temp(1,pat) = 1000* tau_prob(pat,word);
            temp(1,pat) = temp(1,pat) * (state*theta_prob{1,2}(:,pat));
            
            
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
            usernn = 1;
            if usernn==1
                rnninput = (probseq(1:t-1,:))';
                rnnoutput = predict(mynested,rnninput);
                temp = rnnoutput(:,end);
                prob = prob.*(temp');
            end
            probseq(t,:) = prob / sum(prob);
            
        end
        
        assign0(pos+1:pos+num_block,:) = probseq;
        pos = pos + num_block;
    end
end

d1 = zeros(size(assign1));
d2 = zeros(size(assign2));
for i=1:nobs
    [~,c1] = max(assign1(i,:));
    [~,c2] = max(assign2(i,:));
    d1(i,c1) = 1;
    d2(i,c2) = 1;
end

% prepare data instance representation for class prediction
assign = horzcat(assign1,assign2);

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

