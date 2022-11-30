
clc;
clear all;

NB = csvread('Data/NB.csv');
EMat = csvread('Data/EMat.csv');
TA = csvread('Data/Time.csv');
FL = csvread("Data/FL.csv");

Nf = 5;
Nch = size(EMat,1)/Nf;

size1 = 50;
size2 = 60;
stride = 5;

npks = 4;
pkmeta = NB;
metaline = 1;

pkrec = zeros(size(NB,1)*npks,3+3*max(NB(:,3)));
for i=1:size(NB,1)
    pkrec((i-1)*npks+1:i*npks,1:3) = repmat(NB(i,1:3),npks,1);
end
metaline2 = 1;


for c=1:1:(size(EMat,1)/Nf)
    img = EMat(c*Nf,1);
    ppl = EMat(c*Nf,2);
    
    temp = (NB(:,1)==img) .* (NB(:,2)==ppl);
    currentNB = NB(temp==1,:);
    Length=currentNB(1,3);
    
    for trunk=1:Length
        shru = csvread(strcat('Data/m-',int2str(img),'-',int2str(ppl),'-',int2str(trunk),'.csv'));
        
        pks = zeros(floor(size1*size2/stride^2),3);
        
        
        i=1;
        for x=1:size1-stride
            for y=1:size2-stride
                window = shru(x:x+stride,y:y+stride);
                pk = max(max( window ));
                [row, col] = find(window==pk);
                row = row(1,1);
                col = col(1,1);
                if row==1 || row==stride+1
                    continue;
                elseif col==1 || col==stride+1
                    continue;
                end
                pks(i,:) = [x+row y+col pk];
                i=i+1;
            end
        end
        
        pks = unique(pks,'rows');
        pks = sortrows(pks,3,'descend');
        if(pks(end,1)==0)
            pks = pks(1:end-1,:);
        end
        if size(pks,1)>npks
            pks = pks(1:npks,:);
        end
        
        centerx = floor((pks(:,1))'*pks(:,3)  /sum(pks(:,3)));
        centery = floor((pks(:,2))'*pks(:,3)  /sum(pks(:,3)));
        
        Mbi = zeros(size1*3,size2*3);
        Mbi(size1+1:2*size1,size2+1:2*size2) = shru;
        
        shft = Mbi(size1-24+centerx:size1+25+centerx, size2-29+centery:size2+30+centery);
        shft = floor(255*shft/max(max(shft)));
        
        
        newpks = zeros(size1*size2,3);
        i=1;
        for x=1:2:size1-stride
            for y=1:2:size2-stride
                window = shft(x:x+stride,y:y+stride);
                pk = max(max( window ));
                [row, col] = find(window==pk);
                row = row(1,1);
                col = col(1,1);
                
                if row==1 || row==stride+1
                    continue;
                elseif col==1 || col==stride+1
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
        
        temp = size(newpks,1);
        pkmeta(metaline,3+trunk) = temp;
        
        pkrec(metaline2:metaline2+temp-1,4+3*(trunk-1):3+3*trunk) = newpks;
        
    end
    metaline = metaline+1;
    metaline2 = metaline2 + npks;
end
