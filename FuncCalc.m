
clear all;
rng(0);

NB = csvread('Data/NBlock.csv');
EMat = csvread('Data/EMat.csv');
PMeta = csvread('Data/PMeta.csv');
PRec = csvread('Data/PRecord.csv');

Nf = 5;
npks = 4;


size1 = 50;
size2 = 60;
Nclu = 50;
maxIter = 50;


vlook = zeros(256,4);
idx = 1;

for c=1:1:(size(EMat,1)/Nf)
    img = EMat(c*Nf,1);
    ppl = EMat(c*Nf,2);
    temp = (NB(:,1)==img).*(NB(:,2)==ppl);
    ntru = NB(temp==1,3);
    for trunk = 1:ntru
        vlook(idx,:) = [img ppl trunk idx];

        idx = idx+1;
    end
end
vlook = vlook(1:idx-1,:);
nmp = idx-1; 
mps = zeros(size1,size2,nmp);


idx = 1;

for c=1:1:(size(EMat,1)/Nf)
    img = EMat(c*Nf,1);
    ppl = EMat(c*Nf,2);
    temp = (NB(:,1)==img).*(NB(:,2)==ppl);
    ntru = NB(temp==1,3);
    for trunk = 1:ntru

        mps(:,:,idx) = csvread(strcat('Data/s-',int2str(img),'-',int2str(ppl),'-',int2str(trunk),'.csv'));
        idx = idx+1;
    end
end


muq = zeros(size1,size2,Nclu);
mupk = zeros(Nclu*npks,3);
mumeta = zeros(1,Nclu);

for i=1:Nclu

    pick = floor(1+rand()*(nmp-0.1));
    muq(:,:,i) = mps(:,:,pick);
    

    temp = find(vlook(:,4)==pick);
    img = vlook(temp,1);
    ppl = vlook(temp,2);
    trunk = vlook(temp,3);
    
    mumeta(1,i) = PMeta(1,3+trunk);
    temp = (PRec(:,1)==img).*(PRec(:,2)==ppl);
    recnow = PRec(temp==1,:);
    mupk((i-1)*npks+1:i*npks,:) = recnow(:,3+(trunk-1)*3+1:3+trunk*3);
end

z = zeros(Nclu,nmp);
zunn = zeros(Nclu,nmp);
asgn0 = zeros(Nclu,nmp);

storage = cell(Nclu,nmp);
w1store = zeros(2,nmp*Nclu);
w2store = zeros(2,nmp*Nclu);
bestagls = 100*ones(Nclu,nmp);


hard = 0;
for iter=1:maxIter
    
    for idxmp=1:nmp
        for idxclu = 1:Nclu
            temp = find(vlook(:,4)==idxmp);
            img = vlook(temp,1);
            ppl = vlook(temp,2);
            trunk = vlook(temp,3);
            
            temp = (PMeta(:,1)==img).*(PMeta(:,2)==ppl);
            currentmeta = PMeta(temp==1,:);

            n1 = PMeta(1,currentmeta(1,3+trunk));

            n2 = mumeta(1,idxclu);
            minpk = min(n1,n2);
            
            if minpk == 1
                w1 = [1;0];
                w2 = [0;1];
            else

                temp = (PRec(:,1)==img).*(PRec(:,2)==ppl);
                recnow = PRec(temp==1,:);
                originpk = recnow(1:minpk,4+(trunk-1)*3:2+trunk*3);

                temp = zeros(1,81);
                originpk(:,1) = originpk(:,1)-25;
                originpk(:,2) = originpk(:,2)-30;
                for i=1:81
                    agl = -pi+(i-1)/40*pi;
                    w1 = [cos(agl);-sin(agl)];
                    w2 = [sin(agl);cos(agl)];
                    originxy = mupk((idxclu-1)*npks+1:(idxclu-1)*npks+minpk,1:2);
                    originxy(:,1) = originxy(:,1)-25;
                    originxy(:,2) = originxy(:,2)-30;
                    newxy = originxy*[w1,w2];
                    temp(1,i) = sum(sum( (newxy-originpk).^2 ));
                end
                idxbest = find(temp==min(temp));
                idxbest = idxbest(1,1);
                agl = -pi+(idxbest-1)/40*pi;
                w1 = [cos(agl);-sin(agl)];
                w2 = [sin(agl);cos(agl)];
                bestagls(idxclu,idxmp) = idxbest;
            end

            w1store(:,(idxmp-1)*Nclu+idxclu) = w1;
            w2store(:,(idxmp-1)*Nclu+idxclu) = w2;
            

            newmp = zeros(size1,size2); 
            mpnow = mps(:,:,idxmp);
            
            for x=1:size1
                for y=1:size2
                    oldx = round([x-25 y-30]*w1)+25;
                    oldy = round([x-25 y-30]*w2)+30;
                    if oldx<1 || oldx>=size1
                        continue;
                    elseif oldy<1 || oldy>=size2
                        continue;
                    else
                        x0 = floor(oldx);
                        y0 = floor(oldy);
                        newmp(x,y) = (oldx-x0)*(oldy-y0)*mpnow(x0+1,y0+1)+(x0+1-oldx)*(y0+1-oldy)*mpnow(x0,y0)...
                            + (oldx-x0)*(y0+1-oldy)*mpnow(x0+1,y0)+(x0+1-oldx)*(oldy-y0)*mpnow(x0,y0+1);
                    end
                end
            end
            newmp = 255*newmp/max(max(newmp));

            storage{idxclu, idxmp} = newmp;
            
            mu_small = (muq(:,:,idxclu));
            mu_small = mu_small(13:37,16:45);
            newmp_small = newmp(13:37,16:45);
            
            zunn(idxclu, idxmp) = -10/250000*sum(sum((newmp_small-mu_small).^2));
        end
    end
    
    for idxmp = 1:nmp

        [temp, nowcluster] = max(zunn(:,idxmp));
        zunn(:, idxmp) = zunn(:, idxmp)-temp;
        z(:, idxmp) = exp(zunn(:, idxmp)) / sum(exp(zunn(:, idxmp)));

        asgn0(:, idxmp) = zeros(Nclu,1);
        asgn0(nowcluster, idxmp) = 1;
    end
    

    for idxclu = 1:Nclu
        temp = zeros(size1,size2);
        
        if hard==1
            for idxmp = 1:nmp

                temp = temp+asgn0(idxclu,idxmp)*storage{idxclu, idxmp};
            end

            if sum(asgn0(idxclu,:))>0
                temp = temp / sum(asgn0(idxclu,:));
            else 
                pick = floor(1+rand()*(nmp-0.1));
                temp = mps(:,:,pick);
            end
        else
            for idxmp = 1:nmp

                temp = temp+z(idxclu,idxmp)*storage{idxclu, idxmp};
            end
            if sum(asgn0(idxclu,:))>0
                temp = temp / sum(z(idxclu,:));
            else
                pick = floor(1+rand()*(nmp-0.1));
                temp = mps(:,:,pick);
            end
        end

        muq(:,:,idxclu) = temp;
        
        newpks = zeros(size1*size2,3);
        stride = 16;
        i=1;
        for x=1:8:size1-stride
            for y=1:8:size2-stride
                window = temp(x:x+stride,y:y+stride);
                pk = max(max( window ));
                [row, col] = find(window==pk);
                row = row(1,1);
                col = col(1,1);
                if row==1 || row==stride+1
                    continue;
                elseif col==1 || col==stride+1
                    continue;
                elseif pk==0
                    continue;
                end
                newpks(i,:) = [x+row y+col pk];
                i=i+1;
            end
        end
        newpks = unique(newpks,'rows');
        newpks = sortrows(newpks,3,'descend');
        
        if size(newpks,1)>npks
            newpks = newpks(1:npks,:);
        end
        for i=1:npks-1
            if(newpks(end,3)<=16) || (newpks(end,1)==0)
                newpks = newpks(1:end-1,:);
            end
        end
        
        temp2 = size(newpks,1);
        mumeta(1,idxclu) = temp2;
        mupk((idxclu-1)*npks+1:(idxclu-1)*npks+temp2,:) = newpks;
        mupk((idxclu-1)*npks+temp2+1:idxclu*npks,:) = zeros(npks-temp2,3);
    end
    
    
end
